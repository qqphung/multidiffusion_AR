import os
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from stable_diffusion.gaussian_smoothing import GaussianSmoothing
import math
import torch
from torch.nn import functional as F
from torchvision.utils import save_image

def plot_spatial_maps(spatial_map_list, map_names, save_dir, seed, tokens_vis=None):
    ########
    # spa_map: [B x 4 x H x W]
    for i, (spa_map, map_name) in enumerate(zip(spatial_map_list, map_names)):
        n_obj = len(spa_map)
        plt.figure()
        plt.clf()

        fig, axs = plt.subplots(ncols=n_obj+1, gridspec_kw=dict(width_ratios=[1 for _ in range(n_obj)]+[0.1]))

        fig.set_figheight(3)
        fig.set_figwidth(3*n_obj+0.1)

        cmap = plt.get_cmap('YlOrRd')

        vmax = 0
        vmin = 1
        for tid in range(n_obj):
            spatial_map_cur = spa_map[tid]
            spatial_map_cur = spatial_map_cur[0, 0].cpu()
            vmax = max(vmax, float(spatial_map_cur.max()))
            vmin = min(vmin, float(spatial_map_cur.min()))

        for tid in range(n_obj):
            spatial_map_cur = spa_map[tid]
            spatial_map_cur = spatial_map_cur[0, 0].cpu()
            sns.heatmap(
                spatial_map_cur, annot=False, cbar=False, ax=axs[tid],
                cmap=cmap, vmin=vmin, vmax=vmax
            )
            if tokens_vis is not None:
                axs[tid].set_xlabel(tokens_vis[tid])


        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, cax=axs[-1])

        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, 'average_seed%d_spa%d_%s.png' % (seed, i, map_name)), dpi=100)
        plt.close('all')

def get_all_attention(attn_maps_mid, attn_maps_up , attn_maps_down, res):
    result  = []
    # import pdb; pdb.set_trace()
    for attn_map_integrated in attn_maps_up:
        if attn_map_integrated == []: continue
        for attn_map in attn_map_integrated:
            b, i, j = attn_map.shape
            H = W = int(math.sqrt(i))
            # print(H)
            if H == res:
                result.append(attn_map.reshape(-1, res, res,attn_map.shape[-1] ))
    for attn_map_integrated in attn_maps_mid:

    # for attn_map_integrated in attn_maps_mid:
        attn_map = attn_map_integrated
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        # print(H)
        if (H==res):
            result.append(attn_map.reshape(-1, res, res,attn_map.shape[-1] ))
    # import pdb; pdb.set_trace()
    for attn_map_integrated in attn_maps_down:
        if attn_map_integrated == []: continue
        for attn_map in attn_map_integrated:
            if attn_map == []: continue
            # print(attn_map.shape)
            # import pdb; pdb.set_trace()
            b, i, j = attn_map.shape
            H = W = int(math.sqrt(i))
            # print(H)
            if (H==res):
                result.append(attn_map.reshape(-1, res, res,attn_map.shape[-1] ))
    
    result = torch.cat(result, dim=0)
    result = result.sum(0) / result.shape[0]
    # print('shape', result.shape)
    return result

def caculate_loss_att_fixed_cnt(attn_maps_mid, attn_maps_up, attn_maps_down, bboxes_ori, object_positions_ori, t, res=16, smooth_att = True,sigma=0.5,kernel_size=3 ):
    attn16 = get_all_attention(attn_maps_mid, attn_maps_up, attn_maps_down, res)
    attn32 = get_all_attention(attn_maps_mid, attn_maps_up, attn_maps_down, 32)
    attn64 = get_all_attention(attn_maps_mid, attn_maps_up, attn_maps_down, 64)
    attn8 = get_all_attention(attn_maps_mid, attn_maps_up, attn_maps_down, 8)
    all_attn = [attn32, attn16,attn8, attn64]
    
    if len(bboxes_ori) > 6:
        bboxes = bboxes_ori[:6]
        object_positions = object_positions_ori[:6]
    else:
        bboxes = bboxes_ori
        object_positions = object_positions_ori
    obj_number = len(bboxes)
    total_loss = 0
    # import pdb; pdb.set_trace()
    for attn in all_attn[1:2]:
        attn_text = attn[:, :, 1:-1]
        attn_text *= 100
        attn_text = torch.nn.functional.softmax(attn_text, dim=-1)
        current_res =  attn.shape[0]
        H = W = current_res
        
        # if t == 29:  import pdb; pdb.set_trace()
        
        for obj_idx in range(obj_number):
            min_inside =10
            for obj_position in object_positions[obj_idx]:
                true_obj_position = obj_position - 1
                att_map_obj = attn_text[:,:, true_obj_position]
                if smooth_att:
                    smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                    input = F.pad(att_map_obj.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                    att_map_obj = smoothing(input).squeeze(0).squeeze(0)
                other_att_map_obj = att_map_obj.clone()
                att_copy = att_map_obj.clone()

                for obj_box in bboxes[obj_idx]:
                    x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                    int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                
                
                    if att_map_obj[y_min: y_max, x_min: x_max].numel() == 0: 
                        '''if (y_min==y_max) and (x_min==x_max):
                            max_inside = att_map_obj[y_min-1: y_max, x_min-1: x_max].max()
                        if y_min==y_max:
                            max_inside = att_map_obj[y_min-1: y_max, x_min: x_max].max()
                        elif x_min==x_max:
                            max_inside = att_map_obj[y_min: y_max, x_min-1: x_max].max()'''
                        max_inside=1.
                        
                    else:
                        max_inside = att_map_obj[y_min: y_max, x_min: x_max].max()
                    # min_inside = min(max_inside, min_inside)
                    total_loss += 1. - max_inside
                    
                    # find max outside the box, find in the other boxes
                    
                    att_copy[y_min: y_max, x_min: x_max] = 0.
                    other_att_map_obj[y_min: y_max, x_min: x_max] = 0.
                
                max_outside = 0
                for obj_outside in range(obj_number):
                    if obj_outside != obj_idx:
                        for obj_out_box in bboxes[obj_outside]:
                            x_min_out, y_min_out, x_max_out, y_max_out = int(obj_out_box[0] * W), \
                                int(obj_out_box[1] * H), int(obj_out_box[2] * W), int(obj_out_box[3] * H)
                            
                            # att_copy[y_min: y_max, x_min: x_max] = 0.
                            if other_att_map_obj[y_min_out: y_max_out, x_min_out: x_max_out].numel() == 0: 
                                max_outside_one= 0
                                '''if (y_min_out== y_max_out) and (x_min_out== x_max_out) :
                                    max_outside_one = other_att_map_obj[y_min_out-1: y_max_out, x_min_out-1: x_max_out].max() 
                                elif y_min_out== y_max_out:
                                    max_outside_one = other_att_map_obj[y_min_out-1: y_max_out, x_min_out: x_max_out].max() 
                                elif x_min_out== x_max_out and  :
                                    max_outside_one = other_att_map_obj[y_min_out: y_max_out, x_min_out-1: x_max_out].max() ''' 
                                
                            # print(y_min_out,y_max_out, x_min_out, x_max_out )
                            else:
                                max_outside_one = other_att_map_obj[y_min_out: y_max_out, x_min_out: x_max_out].max()
                            # max_outside = max(max_outside,max_outside_one )
                            att_copy[y_min_out: y_max_out, x_min_out: x_max_out] = 0.
                            total_loss += max_outside_one
                max_background = att_copy.max()
                total_loss += len(bboxes[obj_idx]) *max_background /2.
                
    return total_loss/obj_number

def loss_one_att_outside(attn_map,bboxes, object_positions,t):
    # loss = torch.tensor(0).to('cuda')
    loss = 0
    object_number = len(bboxes)
    b, i, j = attn_map.shape
    H = W = int(math.sqrt(i))
    
    import pdb; pdb.set_trace()
    # if t== 20: import pdb; pdb.set_trace()

    for obj_idx in range(object_number):
        
        for obj_box in bboxes[obj_idx]:
            mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
            x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
            mask[y_min: y_max, x_min: x_max] = 1.
            mask = 1. - mask
            index = (mask == 1.).nonzero(as_tuple=False)
            index_out_key = index[:,0]* H + index[:, 1]
            att_box = torch.zeros_like(attn_map)
            att_box[:,index_out_key,:] = attn_map[:,index_out_key,:]

            att_box = att_box.sum(axis=1) / index_in_key.shape[0]
            att_box = att_box.reshape(-1, H, H)
            # activation_value = 
            # smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
            # input = F.pad(att_map_obj.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            # att_map_obj = smoothing(input).squeeze(0).squeeze(0)
            activation_value = (att_box* mask).reshape(b, -1).sum(dim=-1) #/ att_box.reshape(b, -1).sum(dim=-1)
            loss += torch.mean(activation_value)
    return loss


        


def loss_one_att_outside(attn_map,bboxes, object_positions,t):
    # loss = torch.tensor(0).to('cuda')
    loss = 0
    object_number = len(bboxes)
    b, i, j = attn_map.shape
    H = W = int(math.sqrt(i))
    
    
    # if t== 20: import pdb; pdb.set_trace()
    
    for obj_idx in range(object_number):
        
        for obj_box in bboxes[obj_idx]:
            mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
            x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
            mask[y_min: y_max, x_min: x_max] = 1.
            mask_out = 1. - mask
            index = (mask == 1.).nonzero(as_tuple=False)
            index_in_key = index[:,0]* H + index[:, 1]
            att_box = torch.zeros_like(attn_map)
            att_box[:,index_in_key,:] = attn_map[:,index_in_key,:]

            att_box = att_box.sum(axis=1) / index_in_key.shape[0]
            att_box = att_box.reshape(-1, H, H)
            # activation_value = 
            # smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
            # input = F.pad(att_map_obj.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            # att_map_obj = smoothing(input).squeeze(0).squeeze(0)
            activation_value = (att_box* mask_out).reshape(b, -1).sum(dim=-1) #/ att_box.reshape(b, -1).sum(dim=-1)
            loss += torch.mean(activation_value)
            
    return loss / object_number

def caculate_loss_self_att(self_first, self_second, self_third, bboxes, object_positions, t, list_res=[256], smooth_att = True,sigma=0.5,kernel_size=3 ):
    all_attn = get_all_self_att(self_first, self_second, self_third)
    # import pdb; pdb.set_trace()
    # attn_map = torch.cat(attn[res], dim=0)
    # attn_map = attn_map.sum(0) / result.shape[0]
    cnt = 0
    total_loss = 0
    # if t== 10: import pdb; pdb.set_trace()
    for res in list_res:
        attn_maps = all_attn[res]
        for attn in attn_maps:
            total_loss += loss_one_att_outside(attn, bboxes, object_positions,t)
            cnt += 1

    return total_loss /cnt


def get_all_self_att(self_first, self_second, self_third):
    result = {256:[], 1024:[], 4096:[], 64:[], 94:[],1054:[] ,286:[],4126:[] }
    # import pdb; pdb.set_trace()
    all_att = [[self_first], self_second, self_third]
    # import pdb; pdb.set_trace()
    for self_att in all_att:
        for att in self_att:
            if att != []:
                
                for attn_map in att:
                    current_res = attn_map.shape[1]
                    # print(current_res)
                    result[current_res].append(attn_map)
    return result