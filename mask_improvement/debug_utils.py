import os
os.chdir("/workspace")
print(os.getcwd())
import sys
sys.path.append("/workspace")
from CasMVSNet_pl.datasets import DTUDataset
from CasMVSNet_pl.datasets.utils import read_pfm
import cv2
import copy
import numpy as np
from PIL import Image
import torch
import cv2
import numpy as np

import cv2
import numpy as np


import os


import torch

import argparse
#from src.dataset import MP3Ddataset, Scannetdataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from src.lightning_pano_gen import PanoGenerator
from src.lightning_depth import DepthGenerator
#from src.lightning_pano_outpaint import PanoOutpaintGenerator
import pytorch_lightning as pl
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
import torch
import os
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.models.pano.MVGenModel import MultiViewBaseModel
from einops import rearrange
from src.lightning_depth_dreamer import DepthDreamerGenerator

def findhomography(src_img,target_img):
    # Read the source and target images


    # Perform feature detection and matching (using ORB as an example)
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(src_img, None)
    keypoints2, descriptors2 = orb.detectAndCompute(target_img, None)

    # Use a matcher (e.g., BFMatcher) to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches based on their distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract corresponding points
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Use RANSAC to estimate the transformation matrix
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp the source image to the target image
    result_img = cv2.warpPerspective(src_img, M, (target_img.shape[1], target_img.shape[0]))
    return result_img, M
    # Now, 'result_img' contains the source image warped to align with the target image
class DTU(DTUDataset):
    def __init__(self,len,**kwargs):
        super().__init__(**kwargs)
        self.len = len
    def __len__(self):
        return self.len
    def add_shadow(self,image, shadow_intensity=0.7):
        # Read the image
        original_image = copy.deepcopy(image)


        # Create a blank image with the same size as the original image
        shadow_image = np.zeros_like(original_image)

        # Define the shadow color (you can adjust these values as needed)
        shadow_color = (50, 50, 50)

        # Add the shadow to the image
        shadow_image[:, :] = shadow_color
        cv2.addWeighted(shadow_image, shadow_intensity, original_image, 1 - shadow_intensity, 0, original_image)
        return original_image

    def build_proj_mats(self):
        proj_mats = []
        Rs = []
        Ks = []
        for vid in range(49): # total 49 view ids
            if self.img_wh is None:
                proj_mat_filename = os.path.join(self.root_dir,
                                                 f'Cameras/train/{vid:08d}_cam.txt')
            else:
                proj_mat_filename = os.path.join(self.root_dir,
                                                 f'Cameras/{vid:08d}_cam.txt')
            intrinsics, extrinsics, depth_min = \
                self.read_cam_file(proj_mat_filename)
            if self.img_wh is not None: # resize the intrinsics to the coarsest level
                intrinsics[0] *= self.img_wh[0]/1600/4
                intrinsics[1] *= self.img_wh[1]/1200/4

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_ls = []
            Ks.append(intrinsics)
            Rs.append(extrinsics)
            for l in reversed(range(self.levels)):
                proj_mat_l = np.eye(4)
                proj_mat_l[:3, :4] = intrinsics @ extrinsics[:3, :4]
                intrinsics[:2] *= 2 # 1/4->1/2->1
                proj_mat_ls += [torch.FloatTensor(proj_mat_l)]
            # (self.levels, 4, 4) from fine to coarse
            proj_mat_ls = torch.stack(proj_mat_ls[::-1])
            proj_mats += [(proj_mat_ls, depth_min)]
        self.R = Rs
        self.K = Ks
        self.proj_mats = proj_mats

    def build_metas(self):
        self.metas = []
        with open(f'/workspace/CasMVSNet_pl/datasets/lists/dtu/{self.split}.txt') as f:
            self.scans = [line.rstrip() for line in f.readlines()]

        # light conditions 0-6 for training
        # light condition 3 for testing (the brightest?)
        light_idxs = [3] if self.img_wh else range(7)

        pair_file = "Cameras/pair.txt"
        for scan in self.scans:
            with open(os.path.join(self.root_dir, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    for light_idx in light_idxs:
                        self.metas += [(scan, light_idx, ref_view, src_views)]
    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32) # (1200, 1600)
        if self.img_wh is None:
            depth = cv2.resize(depth, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST) # (600, 800)
            depth_0 = depth[44:556, 80:720] # (512, 640)
        else:
            depth_0 = cv2.resize(depth, self.img_wh,
                                 interpolation=cv2.INTER_NEAREST)


        return depth_0
    def extract_boundary(self,img, percentage_threshold=0.3):


        # Apply GaussianBlur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        intensity_range = np.max(img) - np.min(img)
        dynamic_threshold = np.min(img) + percentage_threshold * intensity_range
        # Apply Canny edge detector
        #edges = cv2.Canny(blurred, dynamic_threshold, np.min(img) + 5*percentage_threshold * intensity_range)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_img, dynamic_threshold, 255, cv2.THRESH_BINARY)

        # Create a three-channel image to represent the edges
        #edges_color = cv2.merge([edges, edges, edges])

        # Combine the original image with the edges
        #plt.imshow(edges)
        #plt.show()
        #result = cv2.bitwise_and(img, edges)
        return binary_image

    def read_mask(self, filename):
        mask = cv2.imread(filename, 0) # (1200, 1600)
        #mask = cv2.resize(mask,(1200,1600))
        if self.img_wh is None:
            mask = cv2.resize(mask, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST) # (600, 800)
            mask_0 = mask[44:556, 80:720] # (512, 640)
        else:
            mask_0 = cv2.resize(mask, self.img_wh,
                                interpolation=cv2.INTER_NEAREST)
        mask_1 = cv2.resize(mask_0, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)
        mask_2 = cv2.resize(mask_1, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)


        return mask_0
    def __getitem__(self,idx):

        sample = {}
        scan, light_idx, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs = []
        dark_masks =[]
        prompts = []
        dark_imgs = []
        proj_mats = [] # record proj mats between views
        Rs = []
        Ks = []
        dark_images = []
        proj_mats = []
        masks = []


        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            if self.img_wh is None:
                img_filename = os.path.join(self.root_dir,
                                f'Rectified/{scan}_train/rect_{vid+1:03d}_{light_idx}_r5000.png')
                mask_filename = os.path.join(self.root_dir,
                                f'Depths/{scan}/depth_visual_{vid:04d}.png')
                depth_filename = os.path.join(self.root_dir,
                                f'Depths/{scan}/depth_map_{vid:04d}.pfm')
                prompts_file = os.path.join(self.root_dir,
                                            f'prompts/{scan}_train_prompt/rect_{1:03d}_{3}_r5000.png.txt')

            else:
                img_filename = os.path.join(self.root_dir,
                                f'Rectified/{scan}/rect_{vid+1:03d}_{light_idx}_r5000.png')

            img = Image.open(img_filename)
            img = np.array(img)

            dark_img = self.add_shadow(np.array(img))
            img_mask = self.extract_boundary(np.array(dark_img))
            dark_masks.append(torch.tensor(img_mask))
            dark_images.append(dark_img)
            dark_img = self.transform(dark_img)
            dark_imgs.append(dark_img)

            if self.img_wh is not None:
                img = img.resize(self.img_wh, Image.BILINEAR)
            #img = self.transform(img)
            imgs += [img]

            proj_mat_ls, depth_min = self.proj_mats[vid]
            R = self.R[vid][:3,:3]
            K = self.K[vid]
            Rs.append(R)
            Ks.append(K)
            with open(prompts_file,"r") as f:
                prompt = f.readline().strip()+" in van goth sytled"
                prompts.append(prompt)

            masks.append(self.read_mask(mask_filename))
            if i == 0:  # reference view
                sample['init_depth_min'] = torch.FloatTensor([depth_min])
                if self.img_wh is None:
                    #sample['masks'] = self.read_mask(mask_filename)
                    sample['depths'] = self.read_depth(depth_filename)
                ref_proj_inv = torch.inverse(proj_mat_ls)
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]

        dark_masks = torch.stack(dark_masks)
        images = []
        homographys = []
        for i,img in enumerate(imgs):
            image = self.transform(img)
            #image = torch.tensor(img)
            images.append(image)
            homograph = []
            for j in range(len(dark_images)):
                src_img = np.array(img)
                target_img = np.array(imgs[j])

                image,H = findhomography(src_img,target_img)
                homograph.append(H)
            homographys.append(homograph)
        homographys = torch.tensor(homographys)
        sample["homographys"] = homographys

        dark_imgs = np.stack(dark_imgs)
        dark_imgs = torch.tensor(dark_imgs)
        images = torch.stack(images) # views, 3, H, W)
        proj_mats = torch.stack(proj_mats)[:,:,:3] # (V-1, self.levels, 3, 4) from fine to coarse
        
        sample["prompts"] = prompts
        sample["mask"] = dark_masks
        sample["dark_imgs"] = dark_imgs
        sample['imgs'] = images
        sample["images"] = images
        sample['proj_mats'] = proj_mats
        sample['depth_interval'] = torch.FloatTensor([self.depth_interval])
        sample['scan_vid'] = (scan, ref_view)
        sample["R"] = torch.tensor(Rs)
        sample["K"] = torch.tensor(Ks)

        return sample
    
import torch
import torch.nn.functional as F
from src.models.modules.utils import get_x_2d
from einops import rearrange
def get_correspondences(meta, img_h, img_w):
    homographys = meta["homographys"]
    m = homographys.shape[1]

    correspondences=torch.zeros((homographys.shape[0], m, m, img_h, img_w, 2), device=homographys.device)
    for i in range(m):
        for j in range(m):
            homo_l = homographys[:,i,j]


            xyz_l = torch.tensor(get_x_2d(img_h, img_w),
                                device=homographys.device)
            l = xyz_l.shape[-1]
            xyz_l = (
                xyz_l.reshape(-1, 3).T)[None].repeat(homo_l.shape[0], 1, 1)


            xyz_l = homo_l@xyz_l
            l = 1
            #print(homo_l.shape,xyz_l.shape)
            xy_l = (xyz_l[:, :2]/xyz_l[:, 2:]).permute(0,
                                                    2, 1).reshape(-1, l, img_h, img_w, 2)

            correspondences[:,i,j]=xy_l[:,0]
    return correspondences.to("cuda") 
def get_key_value(key_value, xy_l, homo_r, ori_h, ori_w, ori_h_r, query_h):

    b, c, h, w = key_value.shape
    query_scale = ori_h//query_h
    key_scale = ori_h_r//h

    xy_l = xy_l[:, query_scale//2::query_scale,
                query_scale//2::query_scale]/key_scale-0.5

    key_values = []

    xy_proj = []
    kernal_size=3
    for i in range(0-kernal_size//2, 1+kernal_size//2):
        for j in range(0-kernal_size//2, 1+kernal_size//2):
            xy_l_norm = xy_l.clone()
            xy_l_norm[..., 0] = xy_l_norm[..., 0] + i
            xy_l_norm[..., 1] = xy_l_norm[..., 1] + j
            xy_l_rescale = (xy_l_norm+0.5)*key_scale

            xy_proj.append(xy_l_rescale)

            xy_l_norm[..., 0] = xy_l_norm[..., 0]/(w-1)*2-1
            xy_l_norm[..., 1] = xy_l_norm[..., 1]/(h-1)*2-1
            # print(key_value.device,xy_l_norm.device)
            _key_value = F.grid_sample(
                key_value, xy_l_norm, align_corners=True)
            key_values.append(_key_value)

    xy_proj = torch.stack(xy_proj, dim=1)
    mask = (xy_proj[..., 0] > 0)*(xy_proj[..., 0] < ori_w) * \
        (xy_proj[..., 1] > 0)*(xy_proj[..., 1] < ori_h)

    xy_proj_back = torch.cat([xy_proj, torch.ones(
        *xy_proj.shape[:-1], 1, device=xy_proj.device)], dim=-1)
    xy_proj_back = rearrange(xy_proj_back, 'b n h w c -> b c (n h w)')
    xy_proj_back = homo_r@xy_proj_back

    xy_proj_back = rearrange(
        xy_proj_back, 'b c (n h w) -> b n h w c', h=h, w=w)
    xy_proj_back = xy_proj_back[..., :2]/xy_proj_back[..., 2:]

    xy = get_x_2d(ori_w, ori_h)[:, :, :2]
    xy = xy[query_scale//2::query_scale, query_scale//2::query_scale]
    xy = torch.tensor(xy, device=key_value.device).float()[
        None, None]

    xy_rel = (xy_proj_back-xy)/query_scale

    key_values = torch.stack(key_values, dim=1)

    return key_values, xy_rel, mask


def get_query_value(query, key_value, xy_l, homo_r, img_h_l, img_w_l, img_h_r=None, img_w_r=None):
    if img_h_r is None:
        img_h_r = img_h_l
        img_w_r = img_w_l

    b = query.shape[0]
    m = key_value.shape[1]

    key_values = []
    masks = []
    xys = []

    for i in range(m):
        _, _, q_h, q_w = query.shape
        #print("shape is:",key_value.shape,query.shape,homo_r.shape,xy_l.shape)
        _key_value, _xy, _mask = get_key_value(key_value[:, i], xy_l[:, i], homo_r[:, i],
                                               img_h_l, img_w_l, img_w_r, q_h)

        key_values.append(_key_value)
        xys.append(_xy)
        masks.append(_mask)

    key_value = torch.cat(key_values, dim=1)
    xy = torch.cat(xys, dim=1)
    mask = torch.cat(masks, dim=1)

    return query, key_value, xy, mask

import torch
import torch.nn as nn
from einops import rearrange
from src.models.modules.resnet import BasicResNetBlock
from src.models.modules.transformer import BasicTransformerBlock, PosEmbedding
#from src.models.pano.utils import get_query_value

class CPBlock(nn.Module):
    def __init__(self, dim, flag360=False):
        super().__init__()
        self.attn1 = CPAttn(dim, flag360=flag360)
        self.attn2 = CPAttn(dim, flag360=flag360)
        self.resnet = BasicResNetBlock(dim, dim, zero_init=True)

    def forward(self, x, correspondences, img_h, img_w, R, K, m):
        x = self.attn1(x, correspondences, img_h, img_w, R, K, m)
        x = self.attn2(x, correspondences, img_h, img_w, R, K, m)
        x = self.resnet(x)
        return x


class CPAttn(nn.Module):
    def __init__(self, dim, flag360=False):
        super().__init__()
        self.flag360 = flag360
        self.transformer = BasicTransformerBlock(
            dim, dim//32, 32, context_dim=dim)
        self.pe = PosEmbedding(2, dim//4)

    def forward(self, x, correspondences, img_h, img_w, R, K, m,meta):
        b, c, h, w = x.shape
        x = rearrange(x, '(b m) c h w -> b m c h w', m=m)
        outs = []


        for i in range(m):
            indexs = [(i-1+m) % m, (i+1) % m]

            xy_r=correspondences[:, i, indexs]
            xy_l=correspondences[:, indexs, i]

            x_left = x[:, i]
            x_right = x[:, indexs]

            #R_right = R[:, indexs]
            #K_right = K[:, indexs]

            #l = R_right.shape[1]

            #R_left = R[:, i:i+1].repeat(1, l, 1, 1) # 1, l, 3,3
            #K_left = K[:, i:i+1].repeat(1, l, 1, 1)

            #R_left = R_left.reshape(-1, 3, 3)
            #R_right = R_right.reshape(-1, 3, 3)
            #K_left = K_left.reshape(-1, 3, 3)
            #K_right = K_right.reshape(-1, 3, 3)

            #homo_r = (K_left@torch.inverse(R_left) @
            #          R_right@torch.inverse(K_right))
            homo_r = meta["homographys"][:,indexs,i]
            #homo_r = rearrange(homo_r, '(b l) h w -> b l h w', b=xy_r.shape[0])
            #print("homo is:",homo_r.shape,x_left.shape,x_right.shape)
            query, key_value, key_value_xy, mask = get_query_value(
                x_left, x_right, xy_l, homo_r, img_h, img_w)

            key_value_xy = rearrange(key_value_xy, 'b l h w c->(b h w) l c')
            key_value_pe = self.pe(key_value_xy)

            key_value = rearrange(
                key_value, 'b l c h w-> (b h w) l c')
            mask = rearrange(mask, 'b l h w -> (b h w) l')

            key_value = (key_value + key_value_pe)*mask[..., None]

            query = rearrange(query, 'b c h w->(b h w) c')[:, None]
            query_pe = self.pe(torch.zeros(
                query.shape[0], 1, 2, device=query.device))

            out = self.transformer(query, key_value, query_pe=query_pe)

            out = rearrange(out[:, 0], '(b h w) c -> b c h w', h=h, w=w)
            outs.append(out)
        out = torch.stack(outs, dim=1)

        out = rearrange(out, 'b m c h w -> (b m) c h w')

        return out

import torch
import torch.nn as nn

from einops import rearrange
#from src.models.pano.utils import get_correspondences


class MultiViewBaseModel(nn.Module):
    def __init__(self, unet, config):
        super().__init__()

        self.unet = unet
        self.single_image_ft = config['single_image_ft']

        if config['single_image_ft']:
            self.trainable_parameters = [(self.unet.parameters(), 0.01)]
        else:
            self.cp_blocks_encoder = nn.ModuleList()
            for i in range(len(self.unet.down_blocks)):
                self.cp_blocks_encoder.append(CPAttn(
                    self.unet.down_blocks[i].resnets[-1].out_channels, flag360=True))

            self.cp_blocks_mid = CPAttn(
                self.unet.mid_block.resnets[-1].out_channels, flag360=True)

            self.cp_blocks_decoder = nn.ModuleList()
            for i in range(len(self.unet.up_blocks)):
                self.cp_blocks_decoder.append(CPAttn(
                    self.unet.up_blocks[i].resnets[-1].out_channels, flag360=True))

            self.trainable_parameters = [(list(self.cp_blocks_mid.parameters()) + \
                list(self.cp_blocks_decoder.parameters()) + \
                list(self.cp_blocks_encoder.parameters()), 1.0)]

    def forward(self, latents, timestep, prompt_embd, meta):
        K = meta['K']
        R = meta['R']

        b, m, c, h, w = latents.shape
        img_h, img_w = h*8, w*8
        #print("latents: ",latents.shape,img_h,img_w)
        correspondences=get_correspondences(meta, img_h, img_w)

        # bs*m, 4, 64, 64
        hidden_states = rearrange(latents, 'b m c h w -> (b m) c h w')
        prompt_embd = rearrange(prompt_embd, 'b m l c -> (b m) l c')

        # 1. process timesteps

        timestep = timestep.reshape(-1)
        t_emb = self.unet.time_proj(timestep)  # (bs, 320)
        emb = self.unet.time_embedding(t_emb)  # (bs, 1280)

        hidden_states = self.unet.conv_in(
            hidden_states)  # bs*m, 320, 64, 64

        # unet
        #print("hidden_states:,", hidden_states.shape,correspondences.shape)
        # a. downsample
        down_block_res_samples = (hidden_states,)
        for i, downsample_block in enumerate(self.unet.down_blocks):
            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
                for resnet, attn in zip(downsample_block.resnets, downsample_block.attentions):
                    hidden_states = resnet(hidden_states, emb)

                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample

                    down_block_res_samples += (hidden_states,)
            else:
                for resnet in downsample_block.resnets:
                    hidden_states = resnet(hidden_states, emb)
                    down_block_res_samples += (hidden_states,)
            if m > 1 and meta["use_corres"]==True:
                    hidden_states = self.cp_blocks_encoder[i](
                    hidden_states, correspondences, img_h, img_w, R, K, m,meta)

            if downsample_block.downsamplers is not None:
                for downsample in downsample_block.downsamplers:
                    hidden_states = downsample(hidden_states)
                down_block_res_samples += (hidden_states,)

        # b. mid

        hidden_states = self.unet.mid_block.resnets[0](
            hidden_states, emb)

        if m > 1 and meta["use_corres"]==True:
                hidden_states = self.cp_blocks_mid(
                hidden_states, correspondences, img_h, img_w, R, K, m,meta)

        for attn, resnet in zip(self.unet.mid_block.attentions, self.unet.mid_block.resnets[1:]):
            hidden_states = attn(
                hidden_states, encoder_hidden_states=prompt_embd
            ).sample
            hidden_states = resnet(hidden_states, emb)

        h, w = hidden_states.shape[-2:]

        # c. upsample
        for i, upsample_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(
                upsample_block.resnets)]

            if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
                for resnet, attn in zip(upsample_block.resnets, upsample_block.attentions):
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, emb)
                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample
            else:
                for resnet in upsample_block.resnets:
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, emb)
            if m > 1 and meta["use_corres"]==True:
                    hidden_states = self.cp_blocks_decoder[i](
                    hidden_states,correspondences, img_h, img_w, R, K, m,meta)

            if upsample_block.upsamplers is not None:
                for upsample in upsample_block.upsamplers:
                    hidden_states = upsample(hidden_states)

        # 4.post-process
        sample = self.unet.conv_norm_out(hidden_states)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)
        sample = rearrange(sample, '(b m) c h w -> b m c h w', m=m)
        return sample

class PanoOutpaintGenerator(pl.LightningModule):
    def __init__(self, config,torch_dtype = torch.float32):
        super().__init__()

        self.lr = config['train']['lr']
        self.max_epochs = config['train']['max_epochs'] if 'max_epochs' in config['train'] else 0
        self.diff_timestep = config['model']['diff_timestep']
        self.guidance_scale = config['model']['guidance_scale']
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            config['model']['model_id'], subfolder="tokenizer", torch_dtype= torch_dtype)
        self.text_encoder = CLIPTextModel.from_pretrained(
            config['model']['model_id'], subfolder="text_encoder", torch_dtype= torch_dtype)

        self.vae, self.scheduler, unet = self.load_model(
            config['model']['model_id'], torch_dtype= torch_dtype)
        self.mv_base_model = MultiViewBaseModel(
            unet, config['model'])
        self.trainable_params = self.mv_base_model.trainable_parameters

        self.save_hyperparameters()

    def load_model(self, model_id, torch_dtype):
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype= torch_dtype)
        vae.eval()
        scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler", torch_dtype= torch_dtype)
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype= torch_dtype)
        return vae, scheduler, unet

    @torch.no_grad()
    def encode_text(self, text, device):
        text_inputs = self.tokenizer(
            text, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.cuda()
        else:
            attention_mask = None
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), attention_mask=attention_mask)

        return prompt_embeds[0].float(), prompt_embeds[1]

    @torch.no_grad()
    def encode_image(self, x_input, vae):
        b = x_input.shape[0]

        z = vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)

        z = z.sample()

        # use the scaling factor from the vae config
        z = z * vae.config.scaling_factor
        z = z.float()
        return z

    @torch.no_grad()
    def decode_latent(self, latents, vae):
        b, m = latents.shape[0:2]
        latents = (1 / vae.config.scaling_factor * latents)

        images = []
        for j in range(m):
            image = vae.decode(latents[:, j]).sample
            images.append(image)
        image = torch.stack(images, dim=1)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 1, 3, 4, 2).float().numpy()
        image = (image * 255).round().astype('uint8')

        return image

    def configure_optimizers(self):
        param_groups = []
        for params, lr_scale in self.trainable_params:
            param_groups.append({"params": params, "lr": self.lr * lr_scale})
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-7),
            'interval': 'epoch',  # update the learning rate after each epoch
            'name': 'cosine_annealing_lr',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def prepare_mask_latents(
        self, mask, masked_image, height, width
    ):

        mask = torch.nn.functional.interpolate(
            mask, size=(height // 8, width // 8)
        )

        masked_image_latents = self.encode_image(masked_image, self.vae)

        return mask, masked_image_latents

    def training_step(self, batch, batch_idx):
        meta = {
            'K': batch['K'],
            'R': batch['R'],
            'homographys': batch["homographys"],
            'use_corres': batch["use_corres"]

        }

        device = batch['images'].device
        images=batch['images']
        mask_images = batch["dark_imgs"]
        #images=rearrange(images, 'bs m h w c -> bs m c h w')
        #mask_images=rearrange(mask_images, 'bs m h w c -> bs m c h w')
        mask_latnets, masked_image_latents=self.prepare_mask_image(mask_images)

        prompt_embds = []
        for prompt in batch['prompt']:
            prompt_embds.append(self.encode_text(
                prompt, device)[0])
        m=images.shape[1]
        images=rearrange(images, 'bs m c h w -> (bs m) c h w')
        depths = batch['depths']
        latents=self.encode_image(depths, self.vae)
        latents=rearrange(latents, '(bs m) c h w -> bs m c h w', m=m)
        t = torch.randint(0, self.scheduler.num_train_timesteps,
                        (latents.shape[0],), device=latents.device).long()
        prompt_embds = torch.stack(prompt_embds, dim=1)

        noise = torch.randn_like(latents)
        noise_z = self.scheduler.add_noise(latents, noise, t)
        t = t[:, None].repeat(1, latents.shape[1])

        latents_input = torch.cat([noise_z, mask_latnets, masked_image_latents], dim=2)
        denoise = self.mv_base_model(
            latents_input, t, prompt_embds, meta)
        target = noise


        # eps mode
        loss = torch.nn.functional.mse_loss(denoise, target)
        self.log('train_loss', loss)
        return loss

    def gen_cls_free_guide_pair(self, latents, timestep, prompt_embd, batch):
        latents = torch.cat([latents]*2)
        timestep = torch.cat([timestep]*2)

        R = torch.cat([batch['R']]*2)
        K = torch.cat([batch['K']]*2)

        meta = {
            'K': K,
            'R': R,
            'homographys': torch.cat([batch["homographys"]]*2),
            "use_corres": batch["use_corres"]
        }

        return latents, timestep, prompt_embd, meta

   
    def forward_cls_free(self, latents_high_res, _timestep, prompt_embd, batch, model):
        latents, _timestep, _prompt_embd, meta = self.gen_cls_free_guide_pair(
            latents_high_res, _timestep, prompt_embd, batch)
        #print(latents.shape)
        noise_pred = model(
            latents, _timestep, _prompt_embd, meta)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * \
            (noise_pred_text - noise_pred_uncond)

        return noise_pred

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images_pred = self.inference(batch)
        images = ((batch['images']/2+0.5)
                          * 255).cpu().numpy().astype(np.uint8)

        # compute image & save
        if self.trainer.global_rank == 0:
            self.save_image(images_pred, images, batch['prompt'], batch_idx)

    def prepare_mask_image(self, images,batch):
        bs, m, _, h, w = images.shape

        mask=batch["mask"].unsqueeze(2)
        mask=(mask<=0)*1
        mask = mask.float()
        mask = mask.to(images.device)
        #mask[:,0]=batch["masks"]
        masked_image=images*(mask<0.5)


        mask_latnets=[]
        masked_image_latents=[]
        for i in range(m):

            _mask, _masked_image_latent = self.prepare_mask_latents(
                mask[:,i],
                masked_image[:,i],
                #bs,
                h,
                w,
            )
            mask_latnets.append(_mask)
            masked_image_latents.append(_masked_image_latent)
        mask_latnets = torch.stack(mask_latnets, dim=1)
        masked_image_latents = torch.stack(masked_image_latents, dim=1)
        return mask_latnets, masked_image_latents

    @torch.no_grad()
    def inference(self, batch):
        images = batch["dark_imgs"]
        mask_images = batch["dark_imgs"]

        bs, m, _,h, w = images.shape
        #images=rearrange(images, 'bs m h w c -> bs m c h w')
        #mask_images=rearrange(mask_images, 'bs m h w c -> bs m c h w')
        #print(images.shape,mask_images.shape)
        mask_latnets, masked_image_latents=self.prepare_mask_image(mask_images,batch)

        device = images.device

        latents= torch.randn(
            bs, m, 4, h//8, w//8, device=device)

        prompt_embds = []
        for prompt in batch['prompts']:
            prompt = prompt[0]
            print(prompt)
            prompt_embds.append(self.encode_text(
                prompt, device)[0])
        prompt_embds = torch.stack(prompt_embds, dim=1)

        prompt_null = self.encode_text('', device)[0]
        prompt_embd = torch.cat(
            [prompt_null[:, None].repeat(1, m, 1, 1), prompt_embds])

        self.scheduler.set_timesteps(self.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps


        for i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]]*m, dim=1)
            #print(latents.shape,mask_latnets.shape,masked_image_latents.shape)
            latent_model_input = torch.cat([latents, mask_latnets, masked_image_latents], dim=2)

            noise_pred = self.forward_cls_free(
                latent_model_input, _timestep, prompt_embd, batch, self.mv_base_model)

            latents = self.scheduler.step(
                noise_pred, t, latents).prev_sample

        images_pred = self.decode_latent(
            latents, self.vae)

        return images_pred

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images_pred = self.inference(batch)

        images = ((batch['images']/2+0.5)
                          * 255).cpu().numpy().astype(np.uint8)


        scene_id = batch['image_paths'][0].split('/')[2]
        image_id=batch['image_paths'][0].split('/')[-1].split('.')[0].split('_')[0]

        output_dir = batch['resume_dir'][0] if 'resume_dir' in batch else os.path.join(self.logger.log_dir, 'images')
        output_dir=os.path.join(output_dir, "{}_{}".format(scene_id, image_id))

        os.makedirs(output_dir, exist_ok=True)
        for i in range(images.shape[1]):
            path = os.path.join(output_dir, f'{i}.png')
            im = Image.fromarray(images_pred[0, i])
            im.save(path)
            im = Image.fromarray(images[0, i])
            path = os.path.join(output_dir, f'{i}_natural.png')
            im.save(path)
        with open(os.path.join(output_dir, 'prompt.txt'), 'w') as f:
            for p in batch['prompt']:
                f.write(p[0]+'\n')

    @torch.no_grad()
    def save_image(self, images_pred, images, prompt, batch_idx):

        img_dir = os.path.join(self.logger.log_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)

        with open(os.path.join(img_dir, f'{self.global_step}_{batch_idx}.txt'), 'w') as f:
            for p in prompt:
                f.write(p[0]+'\n')
        if images_pred is not None:
            for m_i in range(images_pred.shape[1]):
                im = Image.fromarray(images_pred[0, m_i])
                im.save(os.path.join(
                    img_dir, f'{self.global_step}_{batch_idx}_{m_i}_pred.png'))
                if m_i < images.shape[1]:
                    im = Image.fromarray(
                        images[0, m_i])
                    im.save(os.path.join(
                        img_dir, f'{self.global_step}_{batch_idx}_{m_i}_gt.png'))
