from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from models.unet_2d_condition import UNet2DConditionModel

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import os
import numpy as np
import collections
from functools import partial

class StableDiffusion(nn.Module):
    def __init__(self, device, v2, min_step, max_step):
        super().__init__()
        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '') # remove the last \n!
                print(f'[INFO] loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
        
            self.token = True
            print(f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')
        
        self.device = device
        self.num_train_timesteps = 1000
        self.min_step = int(min_step)
        self.max_step = int(max_step)
        self.clip_gradient = False
        self.v2 = v2

        print(f'[INFO] loading stable diffusion...')
        model_id = 'runwayml/stable-diffusion-v1-5'

        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder='text_encoder').to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet").to(self.device)

        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                       num_train_timesteps=self.num_train_timesteps, skip_prk_steps=True, steps_offset=1)
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        self.masks = []

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]
        if self.v2:
            print("[INFO] using Stable Diffusion v2, loading new text prompt...")
            import requests
            response = requests.post("https://fffiloni-prompt-converter.hf.space/run/predict", json={
            "data": [
                prompt,
                "best", # use the best text converter
            ]}).json()
            prompt = response["data"]
            print(f'[INFO] loaded new text prompt: {prompt}!')

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings


    def get_text_embeds_list(self, prompts):
        # prompt: [list]
        text_embeddings = []
        for prompt in prompts:
            if self.v2:
                print("[INFO] using Stable Diffusion v2, loading new text prompt...")
                import requests
                response = requests.post("https://fffiloni-prompt-converter.hf.space/run/predict", json={
                "data": [
                    prompt,
                    "best", # use the best text converter
                ]}).json()
                prompt = response["data"]
                print(f'[INFO] loaded new text prompt: {prompt}!')

            # Tokenize text and get embeddings
            text_input = self.tokenizer([prompt], padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

            with torch.no_grad():
                text_embeddings.append(self.text_encoder(text_input.input_ids.to(self.device))[0])

        return text_embeddings


    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100):
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        latents = self.encode_imgs(pred_rgb_512)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        w = (1 - self.alphas_cumprod[t])
        grad = w * (noise_pred - noise)

        latents.backward(gradient=grad, retain_graph=True)

        return 0 # dummy loss value

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5,
                        latents=None, bg_aug_end=1000):

        if latents is None:
            latents = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        n_styles = text_embeddings.shape[0]-1
        print(n_styles, len(self.masks))
        assert n_styles == len(self.masks)
        # import pdb; pdb.set_trace()
        with torch.cuda.amp.autocast():#torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                
                # predict the noise residual
                with torch.no_grad():
                    noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[:1],
                                           attention_control_dict={})['sample']
                    noise_pred_text = None
                    for style_i, mask in enumerate(self.masks):
                        if style_i < len(self.masks) - 1:
                            if t > bg_aug_end:
                                rand_rgb = torch.rand([1, 3, 1, 1]).cuda()
                                black_background = torch.ones([1, 3, height, width]).cuda()*rand_rgb
                                black_latent = self.encode_imgs(black_background)
                                noise = torch.randn_like(black_latent)
                                black_latent_noisy = self.scheduler.add_noise(black_latent, noise, t)
                                masked_latent = (mask>0.001) * latents + (mask<0.001) * black_latent_noisy
                            else:
                                masked_latent = latents
                            noise_pred_text_cur = self.unet(masked_latent, t, encoder_hidden_states=text_embeddings[style_i+1:style_i+2])['sample']
                        else:
                            noise_pred_text_cur = self.unet(latents, t, encoder_hidden_states=text_embeddings[style_i+1:style_i+2])['sample']
                        if noise_pred_text is None:
                            noise_pred_text = noise_pred_text_cur * mask
                        else:
                            noise_pred_text = noise_pred_text + noise_pred_text_cur*mask

                # perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
                

        return latents

    def produce_latents_bg(self, text_embeddings,text_embeds_original, height=512, width=512, num_inference_steps=50, guidance_scale=7.5,
                        latents=None, bg_aug_end=1000):

        if latents is None:
            latents = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        n_styles = text_embeddings.shape[0]-1
        print(n_styles, len(self.masks))
        # assert n_styles == len(self.masks)
        # import pdb; pdb.set_trace()
        masks = torch.cat(self.masks)
        with torch.cuda.amp.autocast():#torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                latents_all = []
                
                # predict the noise residual
                with torch.no_grad():
                    for style_i, mask in enumerate(self.masks):
                        if style_i < len(self.masks) :
                            if t > bg_aug_end:
                                rand_rgb = torch.rand([1, 3, 1, 1]).cuda()
                                black_background = torch.ones([1, 3, height, width]).cuda()*rand_rgb
                                black_latent = self.encode_imgs(black_background)
                                noise = torch.randn_like(black_latent)
                                black_latent_noisy = self.scheduler.add_noise(black_latent, noise, t)
                                masked_latent = (mask>0.001) * latents + (mask<0.001) * black_latent_noisy
                            else:
                                masked_latent = latents
                            latents_all.append(masked_latent)
                        else:
                            latents_all.append(latents)
                    latent_model_input = torch.cat(latents_all * 2)
                    if t > bg_aug_end:
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
                    else:
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds_original)['sample']
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                # perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1

                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
                latents = (latents*masks).sum(dim=0, keepdims=True)
               
                #torch.cuda.empty_cache()

        return latents

    def predict_x0(self, x_t, eps_t, t):
        alpha_t = self.scheduler.alphas_cumprod[t]
        return (x_t - eps_t * torch.sqrt(1-alpha_t)) / torch.sqrt(alpha_t)


    def produce_attn_maps(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50,
                      guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeddings = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]
        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts,prompts_original, negative_prompts='', height=512, width=512, num_inference_steps=50,
                      guidance_scale=7.5, latents=None, bg_aug_end=1000):

        if isinstance(prompts, str):
            prompts = [prompts]
            prompts_original = [prompts_original]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]
        text_embeds_original = self.get_text_embeds(prompts_original, negative_prompts)
        
        latents = self.produce_latents_bg(text_embeds,text_embeds_original,  height=height, width=width, latents=latents,
                                    num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                                    bg_aug_end=bg_aug_end) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs

    def reset_attention_maps(self):
            r"""Function to reset attention maps.
            We reset attention maps because we append them while getting hooks
            to visualize attention maps for every step.
            """
            for key in self.attention_maps:
                self.attention_maps[key] = []

    def register_evaluation_hooks(self):
        r"""Function for registering hooks during evaluation.
        We mainly store activation maps averaged over queries.
        """
        self.forward_hooks = []
        def save_activations(activations, name, module, inp, out):
            r"""
            PyTorch Forward hook to save outputs at each forward pass.
            """
            # out[0] - final output of attention layer
            # out[1] - attention probability matrix
            if 'attn2' in name:
                assert out[1].shape[-1] == 77
                activations[name].append(out[1].detach().cpu())
            else:
                assert out[1].shape[-1] != 77
        attention_dict = collections.defaultdict(list)
        for name, module in self.unet.named_modules():
            leaf_name = name.split('.')[-1]
            if 'attn' in leaf_name:
                # Register hook to obtain outputs at every attention layer.
                self.forward_hooks.append(module.register_forward_hook(
                    partial(save_activations, attention_dict, name)
                ))
        # attention_dict is a dictionary containing attention maps for every attention layer
        self.attention_maps = attention_dict

    def remove_evaluation_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()
        self.attention_maps = None
