import os
import torch
import random
import imageio
import argparse
import numpy as np
import csv
from stable_diffusion.mixture_of_diffusion_newloss import StableDiffusion
from stable_diffusion.vis_utils import plot_spatial_maps
from chatGPT import generate_box, save_img, load_gt, load_box, draw_box
from PIL import Image
import time
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def refine_boxes(boxes):
    result = []
    for box in boxes:
        result.append((box[1], box[0], box[3], box[2]))
    return result

def format_box(names, boxes):
    result_name = []
    
    for i, name in enumerate(names):
        result_name.append('a ' + name.replace('_',' '))
       
    return result_name, np.array(boxes)

def remove_numbers(text):
    result = ''.join([char for char in text if not char.isdigit()])
    return result
def process_box_phrase(names, bboxes):
    d = {}
    # import pdb; pdb.set_trace()
    for i, phrase in enumerate(names):
        phrase = phrase.replace('_',' ')
        list_noun = phrase.split(' ')
        for n in list_noun:
            n = remove_numbers(n)
            if not n in d.keys():
                d.update({n:[np.array(bboxes[i])/512]})
            else:
                d[n].append(np.array(bboxes[i])/512)
    return d
def Pharse2idx_2(prompt, name_box):
    prompt = prompt.replace('.','')
    prompt = prompt.replace(',','')
    prompt_list = prompt.strip('.').split(' ')
    
    object_positions = []
    bbox_to_self_att = []
    for obj in name_box.keys():
        obj_position = []
        in_prompt = False
        for word in obj.split(' '):
            if word in prompt_list:
                obj_first_index = prompt_list.index(word) + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word +'s' in prompt_list:
                obj_first_index = prompt_list.index(word+'s') + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word +'es' in prompt_list:
                obj_first_index = prompt_list.index(word+'es') + 1
                obj_position.append(obj_first_index)
                in_prompt = True 
            elif word == 'person':
                obj_first_index = prompt_list.index('people') + 1
                obj_position.append(obj_first_index)
                in_prompt = True 
            elif word == 'mouse':
                obj_first_index = prompt_list.index('mice') + 1
                obj_position.append(obj_first_index)
                in_prompt = True 
        if in_prompt :
            bbox_to_self_att.append(np.array(name_box[obj]))
        
            object_positions.append(obj_position)

    return object_positions, bbox_to_self_att
def read_csv(path_file, t):
    list_prompts = []
    with open(path_file,'r') as f:
        reader = csv.reader(f)
        # import pdb; pdb.set_trace()
        for i, row in enumerate(reader):
            if i >0:
                
                if  row[1] ==t:# 'Positional': #row[1] == 'Positional' or
                    list_prompts.append(row[0])
    return list_prompts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters',      type=int, default=10000,            help="training iters")
    parser.add_argument('--text',       type=str, default="A hamburger",    help="text prompt")
    parser.add_argument('--workspace',  type=str, default="test",        help="workspace to store result")
    parser.add_argument('--min_step',   type=int, default=20,               help="min step, range [0, max_step]")
    parser.add_argument('--max_step',   type=int, default=980,              help="max step, range [min_step, 1000]")
    parser.add_argument('--guidance',   type=float, default=100,            help="guidance scale")
    parser.add_argument('--v2',         action='store_true',                help="SD v2.1, default v1.5")
    ### rich text configs
    parser.add_argument('--foldername', type=str, default="test",           help="folder name under workspace")
    parser.add_argument('--seed',       type=int, default=0,                help="random seed")
    parser.add_argument('--type', type=str, default="counting",           help="folder name under workspace")
    parser.add_argument('--data', type=str, default="drawbench",           help="folder name under workspace")
    opt = parser.parse_args()
    seed = opt.seed


    # stable diffusion guidance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    guidance = StableDiffusion(device, opt.v2, opt.min_step, opt.max_step)

    os.makedirs(opt.workspace, exist_ok=True)

    
    save_path = os.path.join(opt.workspace, opt.foldername)
    os.makedirs(save_path, exist_ok=True)


    negative_text = ''
    sum_time = 0 
    # prompts = ['four colorfull easter eggs on the grass']
    # prompts = ['four apples on the table']
    prompts = ['a dog and a cat playing ball on the grass']
    
    for id_img, prompt in enumerate(prompts):    
        print(id_img)
        text = prompt
        # names and box of objects
        # names, boxes =['easter egg', 'easter egg', 'easter egg', 'easter egg'], [(64, 64, 249, 249), (264, 64, 449, 249), (64, 264, 249, 449), (264, 264, 449, 449)]
        names, boxes = ['dog', 'cat', 'ball'], [(78, 123, 288, 388), (298, 185, 478, 390), (220, 350, 372, 450)]
        name_box = process_box_phrase(names, boxes)
        position, box_att = Pharse2idx_2(prompt, name_box)
        boxes = refine_boxes(boxes)
        text_prompts = []
        for name in names:
            text_prompts.append('a ' + name.replace("_"," "))
        #text for background
        new_text = 'on the grass'
        text_copy = text_prompts.copy()
        text_prompts.append(new_text)
        text_copy.append(text)

        print('object', text_prompts)
        print('boxes', boxes)
        height=512
        width=512
        boxes = np.array(boxes)
        
        spatial_map_overall = torch.ones([1, 4,  width//8, height//8]).cuda()
        spatial_maps = []
        for i, o in enumerate(names):

            layout_xy = boxes[i] # [cat, dog]: # 
            latent_layout_xy = [xy//8 for xy in layout_xy]
            spatial_map = torch.zeros([1, 4, width//8, height//8]).cuda()
            spatial_map[:, :, latent_layout_xy[0]:latent_layout_xy[2], latent_layout_xy[1]:latent_layout_xy[3]] = 1.
            spatial_map_overall[:, :, latent_layout_xy[0]:latent_layout_xy[2], latent_layout_xy[1]:latent_layout_xy[3]] = 0.
            spatial_maps.append(spatial_map)

        spatial_maps.append(spatial_map_overall)
        spatial_maps = [spatial_map / torch.cat(spatial_maps, 0).sum(0, keepdim=True) for spatial_map in spatial_maps]
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        
        file_name = text.replace(' ', '_')
        
        assert (torch.cat(spatial_maps, 0).sum(0) == 1).all()
        
        # generate baseline image according to only prompt
        #different random seeds
        for i in range(10):
            print(i)
            seed_everything(i)
            guidance.masks = spatial_maps
            
            bg_aug_end=830 # cang lon cang the prompt cuoi (3 qua trung), cang nho thi cang bi mat background
            start_time = time.time()
            img = guidance.prompt_to_img(text_prompts,text_copy, [negative_text]*len(text_prompts), 
                                        height=height, width=width, num_inference_steps=50, 
                                        guidance_scale=8.5, bg_aug_end=bg_aug_end,locations=position,boxes=box_att )
            if id_img >= 1: sum_time += time.time() - start_time; print('time', sum_time)
            save_img(save_path, Image.fromarray(img[0]), prompt, i, id_img)
        
   #print('sum time', sum_time)
            



# 840/ 61
