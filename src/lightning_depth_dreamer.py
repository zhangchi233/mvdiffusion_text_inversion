import pytorch_lightning as pl
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
import torch
import os
import re
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR
from .models.depth.MVDepthModel import MultiViewBaseModel
#import cv2
# from torch import transformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn


import pytorch_lightning as pl
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
import torch
import os
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR
#import cv2


class DepthDreamerGenerator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config=config

        self.lr = config['train']['lr']
        self.max_epochs = config['train']['max_epochs']
        self.diff_timestep = config['model']['diff_timestep']
        self.guidance_scale = config['model']['guidance_scale']
        self.model_type = config['model']['model_type']

        model_id = config['model']['model_id']
        model_id = "stabilityai/stable-diffusion-2-depth"
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.vae.eval()
        self.scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder")
       
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet")

        self.mv_base_model = MultiViewBaseModel(
            unet, config['model'])
        self.trainable_params = self.mv_base_model.trainable_parameters
        self.save_hyperparameters()

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
        return prompt_embeds
    def encode_post(self,poses,device):
        b,m,c = poses.shape
    @torch.no_grad()
    def encode_image(self, x_input):

        b = x_input.shape[0]

        x_input = x_input.permute(0, 1, 4, 2, 3)  # (bs, 2, 3, 512, 512)
        # (bs*2, 3, 512, 512)
        x_input = x_input.reshape(-1,
                                  x_input.shape[-3], x_input.shape[-2], x_input.shape[-1])
        z = self.vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)
        z = z.sample()
        z = z.reshape(b, -1, z.shape[-3], z.shape[-2],
                      z.shape[-1])  # (bs, 2, 4, 64, 64)

        z = z * 0.18215

        return z

    @torch.no_grad()
    def decode_latent(self, latents):
        b = latents.shape[0]
        latents = latents / 0.18215
        latents = rearrange(latents, 'b m c h w -> (b m) c h w')

        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = rearrange(image, '(b m) c h w -> b m c h w', b=b)
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

    def training_step(self, batch, batch_idx):
        latents_image = self.encode_image(batch['images'])

        prompt_embd = []
        for prompt in batch['prompt']:
            prompt_embd.append(self.encode_text(
                prompt, latents_image.device)[0])
            print("training prompt is ",prompt)

        t = torch.randint(0, self.scheduler.num_train_timesteps,
                          (latents_image.shape[0],), device=latents_image.device).long()

        prompt_embd = torch.stack(prompt_embd, dim=1)
        
        noise = torch.randn_like(latents_image)
        noise_z = self.scheduler.add_noise(latents_image, noise, t)
        t = t[:, None].repeat(1, latents_image.shape[1])
        
        noise_z = torch.cat([noise_z, batch['depth_inv_norm_small'][:,:,None]], dim=2)
        
        b, m, c , h, w=noise_z.shape
        
        mask=torch.zeros((b,m,1,h,w), device=latents_image.device)
        condition=torch.zeros_like(latents_image)
        for i in range(batch['mask'].shape[1]):
            for b in range(batch['mask'].shape[0]):
                if batch['mask'][b,i]:
                    condition[b,i]=latents_image[b,i]
                    mask[b,i]=1

        condition=torch.cat([condition,mask],dim=2)
        batch['condition']=condition
 
        denoise = self.mv_base_model(
            noise_z, t, prompt_embd, batch)
    
        # eps mode
        target = noise
        loss = torch.nn.functional.mse_loss(denoise, target)

        self.log('train_loss', loss)
        return loss

    def gen_cls_free_guide_pair(self, latents, timestep, prompt_embd, batch, type='generation'):
        if type == 'interpolation':
            latents_depth = latents
            torch.cat([latents, batch['depth_inv_norm_small'][:,:,None]], dim=2)
            b, m, c , h, w=latents.shape
            mask=torch.zeros((b,m,1,h,w), device=latents.device)
            condition=torch.zeros_like(latents)

            mask[:,0]=1
            mask[:,-1]=1
            condition[:,0]=batch['images_condition'][:,0]
            condition[:,-1]=batch['images_condition'][:,-1]
            condition=torch.cat([condition,mask],dim=2)
            latents=latents_depth
            meta={
                'condition':torch.cat([condition]*2)
            }
        elif type=='generation':
            depth_input=batch['depth_inv_norm_small'][:,:,None]
            latents = torch.cat([latents, depth_input], dim=2)
            latents = latents
            meta={}
        else:
            raise NotImplementedError
        latents = torch.cat([latents]*2)
        timestep = torch.cat([timestep]*2)
        poses=torch.cat([batch['poses']]*2)
        K=torch.cat([batch['K']]*2)
        depths=torch.cat([batch['depths']]*2)
        meta['poses']=poses
        meta['K']=K
        meta['depths']=depths

        return latents, timestep, prompt_embd, meta

    @torch.no_grad()
    def forward_cls_free(self, latents, _timestep, prompt_embd, batch, model, type):
        _latents, _timestep, _prompt_embd, meta = self.gen_cls_free_guide_pair(
            latents, _timestep, prompt_embd, batch, type)

        noise_pred = model(
            _latents, _timestep, _prompt_embd, meta)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * \
            (noise_pred_text - noise_pred_uncond)

        return noise_pred

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images_pred = self.inference_gen(batch)
       
        images= ((batch['images']/2+0.5)
                          * 255).cpu().numpy().astype(np.uint8)
        
        # compute image & save
        if self.trainer.global_rank == 0:
            self.save_image(images_pred, images, batch['prompt'][0], batch['depth_inv_norm_full'].cpu().numpy(), batch_idx)

    @torch.no_grad()
    def inference_inp(self, batch):
        
        images = batch['images']
        images_latent=self.encode_image(images)

        bs, m, h, w = batch['depths'].shape

        device = images.device

        latents = torch.randn(
            bs, m, 4, h//8, w//8, device=device)

        prompt_embd = []
        for prompt in batch['prompt']:
            prompt_embd.append(self.encode_text(
                prompt[0], latents.device)[0])
        prompt_embd = torch.stack(prompt_embd, dim=1)

        prompt_null = self.encode_text('', device)[0]

        prompt_embd = torch.cat(
            [prompt_null[:, None].repeat(1,  m, 1, 1), prompt_embd])
       
        self.scheduler.set_timesteps(self.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            batch['images_condition']=images_latent

            _timestep = torch.cat([t[None, None]]*m, dim=1)
            noise_pred = self.forward_cls_free(
                latents, _timestep, prompt_embd, batch, self.mv_base_model, type='interpolation')

            latents = self.scheduler.step(
                noise_pred, t, latents).prev_sample
        images_pred = self.decode_latent(latents)
        return images_pred

    @torch.no_grad()
    def inference_gen(self, batch):
        images = batch['images']
        
        bs, m, h, w, _ = images.shape

        device = images.device

        latents= torch.randn(
            bs, m, 4, h//8, w//8, device=device)

        prompt_embd = []
        for prompt in batch['prompt']:
            prompt_embd.append(self.encode_text(
                prompt[0].replace("statue","van goth styled statue"), latents.device)[0])
            print("validation prompt is: ",prompt[0].replace("statue","van goth styled statue"))

        prompt_embd = torch.stack(prompt_embd, dim=1)
        
        prompt_null = self.encode_text('', device)[0]

        prompt_embd = torch.cat(
            [prompt_null[:, None].repeat(1,  m, 1, 1), prompt_embd])
       

        self.scheduler.set_timesteps(self.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]]*m, dim=1)
            noise_pred = self.forward_cls_free(
                latents, _timestep, prompt_embd, batch, self.mv_base_model, type='generation')

            latents = self.scheduler.step(
                noise_pred, t, latents).prev_sample
        images_pred = self.decode_latent(latents)
        return images_pred

    def get_gen_image(self, batch):
        images=batch['images']
        depths=batch['depths']
        poses=batch['poses']
        K=batch['K']
        mask=batch['mask']
        prompt=batch['prompt']
        depth_inv_norm=batch['depths']
        depth_inv_norm = 1. / (depths + 1e-6)
        depth_valid_mask = depths > 0

        depth_inv_norm_small=batch['depth_inv_norm_small']
        
        batch_gen={
            'images': images[0, mask[0]][None],
            'depths': depths[0, mask[0]][None],
            'poses': poses[0, mask[0]][None],
            'K': K,
            'prompt': [p for i, p in enumerate(prompt) if mask[0, i]],
            'depth_inv_norm': depth_inv_norm[0, mask[0]][None],
            'depth_inv_norm_small': depth_inv_norm_small[0, mask[0]][None]
        }
        return batch_gen

    def get_inp_image(self, batch, images_pred):
        key_img_idx=torch.where(batch['mask'][0])[0]
        batches=[]
        images_pred_tensor=torch.tensor(images_pred, device=batch['images'].device)/127.5-1
        for i in range(len(key_img_idx)-1):
            if key_img_idx[i+1]-key_img_idx[i]==1:
                continue
            start_idx=key_img_idx[i]
            end_idx=key_img_idx[i+1]+1
            batch_inp={
                'key_idx': (start_idx, end_idx),
                'images': images_pred_tensor[:,i:i+2],
                'depths': batch['depths'][0, start_idx:end_idx][None],
                'poses': batch['poses'][0, start_idx:end_idx][None],
                'K': batch['K'],
                'prompt': [p for p in batch['prompt'][start_idx:end_idx]],
                'depth_inv_norm': batch['depth_inv_norm'][0, start_idx:end_idx][None],
                'depth_inv_norm_small': batch['depth_inv_norm_small'][0, start_idx:end_idx][None]
            }
            batches.append(batch_inp)
        return batches

    @torch.no_grad()
    def test_step(self, batch, batch_idx):  
        batch_gen=self.get_gen_image(batch)
        images_gen_pred = self.inference_gen(batch_gen)
       
        batches_inp=self.get_inp_image(batch, images_gen_pred)
        images_pred=np.zeros_like(batch['images'].cpu().numpy()).astype(np.uint8)
        mask=batch['mask'][0].cpu().numpy()
        images_pred[0, mask]=images_gen_pred[0]
        for batch_inp in batches_inp:
            images_inp_pred=self.inference_inp(batch_inp)
            idx1=batch_inp['key_idx'][0]
            idx2=batch_inp['key_idx'][1]
            images_pred[0,idx1+1:idx2-1]=images_inp_pred[0,1:-1]

        # compute image & save
        image_paths = batch['image_paths']
        scene_name = image_paths[0][0].split('/')[-3]
        key_id = image_paths[0][0].split('/')[-1].split('.')[0]
        output_dir = os.path.join(
            self.logger.log_dir, 'images', '{}_{}'.format(scene_name, key_id))
        os.makedirs(output_dir, exist_ok=True)
        
        images = ((batch['images']+1)
                          * 127.5).cpu().numpy().astype(np.uint8)
        depths = (batch['depths']*1000).cpu().numpy().astype(np.uint16)
        poses = batch['poses'].cpu().numpy().astype(np.float32)
        K = batch['K'].cpu().numpy().astype(np.float32)[0]
        depth_inv_norm_full=batch['depth_inv_norm'].cpu().numpy().astype(np.float32)
        np.savetxt(os.path.join(output_dir, 'K.txt'), K)
        
        for i, path in enumerate(image_paths):
            path = path[0]
            image_id = path.split('/')[-1].split('.')[0]

            image_pred_path = os.path.join(
                output_dir, '{}_pred.png'.format(image_id))
            Image.fromarray(images_pred[0, i]).save(image_pred_path)

            image_gt_path = os.path.join(
                output_dir, '{}_gt.png'.format(image_id))
            Image.fromarray(images[0, i]).save(image_gt_path)

            depth_gt_path = os.path.join(
                output_dir, '{}_depth.png'.format(image_id))
            cv2.imwrite(depth_gt_path, depths[0, i])

            poses_path = os.path.join(
                output_dir, '{}_poses.txt'.format(image_id))
            np.savetxt(poses_path, poses[0, i])

            plt.imsave(os.path.join(
                 output_dir, '{}_depth_inv.png'.format(image_id)), depth_inv_norm_full[0, i])


    def save_image(self, images_pred, images, prompt, depth_inv_full, batch_idx):

        img_dir = os.path.join(self.logger.log_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        with open(os.path.join(img_dir, f'{self.global_step}_{batch_idx}.txt'), 'w') as f:
            for p in prompt:
                f.write(p)
            
        if images_pred is not None:
            for m_i in range(images_pred.shape[1]):
                im = Image.fromarray(images_pred[0, m_i])
                im.save(os.path.join(
                    img_dir, f'{self.global_step}_{batch_idx}_{m_i}_pred.png'))

        for m_i in range(images.shape[1]):
            im = Image.fromarray(
                images[0, m_i])
            im.save(os.path.join(
                img_dir, f'{self.global_step}_{batch_idx}_{m_i}_gt.png'))
            
            plt.imsave(os.path.join(
                img_dir, f'{self.global_step}_{batch_idx}_{m_i}_depth_inv.png'), depth_inv_full[0, m_i])

    '''
    def __init__(self, config):
        super().__init__()
        self.config=config

        self.lr = config['train']['lr']
        self.max_epochs = config['train']['max_epochs']
        self.diff_timestep = config['model']['diff_timestep']
        self.guidance_scale = config['model']['guidance_scale']
        self.model_type = config['model']['model_type']
        self.place_holder = config['model']['place_holder']
        model_id = config['model']['model_id']
        self.unique_place_holder = config['model']['place_holder']
        torch_type = torch.float32
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae",torch_dtype=torch_type)
        self.vae.eval()
        self.scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler",torch_dtype=torch_type)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer",torch_dtype=torch_type)
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder",torch_dtype=torch_type)

        output_dim = self.text_encoder.config.hidden_size
        camera_embedding = nn.Linear(16, output_dim)
        camera_transformer = TransformerEncoder(TransformerEncoderLayer(d_model=output_dim, nhead=8), num_layers=2)
        camera_sequence = nn.Sequential(camera_embedding, camera_transformer)
        self.camera_encoder = camera_sequence
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet",torch_dtype=torch_type)

        self.mv_base_model = MultiViewBaseModel(
            unet, config['model'])

        
        self.trainable_params = self.mv_base_model.trainable_parameters
        
        # only train text encoder and camera encoder
       
        # self.trainable_params += [
                                 #(list(self.text_encoder.parameters()),1.0),
        #                         (list(self.camera_encoder.parameters()), 1.0),
        #                         (list(self.unet.parameters()),1.0)]
        # frozen vae and unet

        self.save_hyperparameters()

    def encode_post(self,poses,device):
        # poses: bs, m, 4, 4
        # camera_embd: bs, m, prompt_dim
        poses=rearrange(poses, 'b m h w -> b m (h w)')
        camera_embd=self.camera_encoder(poses)
        return camera_embd
   
    def encode_text(self, text, device):
        text_inputs = self.tokenizer(
            text, 
            padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")
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

        x_input = x_input.permute(0, 1, 4, 2, 3)  # (bs, 2, 3, 512, 512)
        x_input = x_input.reshape(-1,
                                  x_input.shape[-3], x_input.shape[-2], x_input.shape[-1])
        z = vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)

        z = z.sample()
        z = z.reshape(b, -1, z.shape[-3], z.shape[-2],
                      z.shape[-1])  # (bs, 2, 4, 64, 64)

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
        return optimizer
        #return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    def training_step(self, batch, batch_idx):      
        T = batch['R'][:, :, :3, 3:]
        R = batch['R'][:, :, :3, :3]
        K = batch['K'][:, :, :3, :3]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #pose = batch["R"]
        #pose_embedding = self.encode_post(pose,device)

        
    

        batch["T"] = T
        batch["R"] = R
        batch["K"] = K
        
        meta = {
            'K': batch['K'].to(device),
            'R': batch['R'].to(device),
            "T": batch['T'].to(device),
            "Depth": batch['depths'].to(device),
        }

        device = batch['images'].device
        prompt_embds = []
        #prompt_pr = [] # prompt has not add identifier
        
        for prompt in batch['prompt']:
            new_sentence = re.sub(r'(a.*?a)\s\w+', r'\1 rabbit', prompt[0], count=1)
            prompt_embds.append(self.encode_text(
                #prompt[0]+"of {0}".format(self.unique_place_holder), 
                new_sentence.replace("a rabbit", "a {0} rabbit".format(self.unique_place_holder)),
                device)[0])
            print(new_sentence.replace("a rabbit", "a {0} rabbit".format(self.unique_place_holder)))
        #    prompt_pr.append(self.encode_text(
        #        new_sentence,device)[0])

        prompt_embds = torch.stack(prompt_embds, dim=1)
        #prompt_pr = torch.stack(prompt_pr, dim=1)
        #pose_embedding = pose_embedding.repeat(1,prompt_embds.shape[1],1,1)
        

        #prompt_embds=torch.cat([prompt_embds,pose_embedding],dim=2)
        #prompt_pr=torch.cat([prompt_pr,pose_embedding],dim=2)


        batch["images"] = batch["images"].to(self.vae.dtype)
        latents = self.encode_image(
            batch['images'], self.vae)
        t = torch.randint(0, self.scheduler.num_train_timesteps,
                        (latents.shape[0],), device=latents.device).long()
        
        
        images_pr = batch['x_pr']
        images_pr = images_pr.to(self.vae.dtype)
        #latents_pr = self.encode_image(
        #    images_pr, self.vae)
        device = images_pr.device
        #noise_pr = torch.randn_like(latents_pr)
        #noise_z_pr = self.scheduler.add_noise(latents_pr, noise_pr, t)
            
        
        
        

        noise = torch.randn_like(latents)
        noise_z = self.scheduler.add_noise(latents, noise, t)
        t = t[:, None].repeat(1, latents.shape[1])
        #noise_z_pr = noise_z_pr.to(self.vae.dtype)
        noise_z = noise_z.to(self.vae.dtype)
        prompt_embds = prompt_embds.to(self.vae.dtype)
        #prompt_pr = prompt_pr.to(self.vae.dtype)
        
        denoise = self.mv_base_model(
            noise_z, t, prompt_embds, meta)
        
        #denoise_pr = self.mv_base_model(noise_z_pr, t, prompt_pr, meta)
        target = noise  
        #target_pr = noise_pr     

        # eps mode
        target = target.to(self.vae.dtype)
        #target_pr = target_pr.to(self.vae.dtype)
        loss1 = torch.nn.functional.mse_loss(denoise, target)
        #loss2 = torch.nn.functional.mse_loss(denoise_pr, target_pr)
        loss = loss1#+0.3*loss2
        self.log('train_loss', loss)
        return loss

    def gen_cls_free_guide_pair(self, latents, timestep, prompt_embd, batch):
        latents = torch.cat([latents]*2)
        timestep = torch.cat([timestep]*2)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        R = torch.cat([batch['R']]*2)[:,:,:3,:3]
        K = torch.cat([batch['K']]*2)[:,:,:3,:3]
        T = torch.cat([batch['R']]*2)[:,:,:3,3:]
        depth = torch.cat([batch['depths']]*2)
        meta = {
            'K': K.to(device),
            'R': batch["R"].to(device),
            "T": T.to(device),
            "Depth": depth.to(device),
        }

        return latents, timestep, prompt_embd, meta

    @torch.no_grad()
    def forward_cls_free(self, latents_high_res, _timestep, prompt_embd, batch, model):
        latents, _timestep, _prompt_embd, meta = self.gen_cls_free_guide_pair(
            latents_high_res, _timestep, prompt_embd, batch)

        noise_pred = model(
            latents, _timestep, _prompt_embd, meta)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * \
            (noise_pred_text - noise_pred_uncond)

        return noise_pred

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        #for prompt in batch["prompt"]:
        #    prompt = ("A van goth style of "+prompt[0]+"of {0}".format(self.unique_place_holder),1.0)
        images_pred = self.inference(batch)
        images = ((batch['images']/2+0.5)
                          * 255).cpu().numpy().astype(np.uint8)
      
        # compute image & save
        if self.trainer.global_rank == 0:
            self.save_image(images_pred, images, batch['prompt'], batch_idx)

    @torch.no_grad()
    def inference(self, batch,prompt_change = True):
        images = batch['images']
        bs, m, h, w, _ = images.shape
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        latents= torch.randn(
            bs, m, 4, h//8, w//8, device=device)

        prompt_embds = []
        for prompt in batch['prompt']:
            new_sentence = re.sub(r'(a.*?a)\s\w+', r'\1 rabbit', prompt[0], count=1)
            if prompt_change:
                changed_sentence = new_sentence.replace("a rabbit", "a {0} rabbit in van goth style".format(self.unique_place_holder))
            else:
                changed_sentence = new_sentence
            print("validation sentence is: ",changed_sentence)
            prompt_embds.append(self.encode_text(
                changed_sentence,
                device)[0])
        
        prompt_embds = torch.stack(prompt_embds, dim=1)

        prompt_null = self.encode_text('', device)[0]
        prompt_embd = torch.cat(
            [prompt_null[:, None].repeat(1, m, 1, 1), prompt_embds])

        self.scheduler.set_timesteps(self.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]]*m, dim=1)

            noise_pred = self.forward_cls_free(
                latents, _timestep, prompt_embd, batch, self.mv_base_model)

            latents = self.scheduler.step(
                noise_pred, t, latents).prev_sample
        latents = latents.to(self.vae.dtype)
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
    '''