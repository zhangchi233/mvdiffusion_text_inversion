import os
import yaml
import pickle
import argparse
import importlib
import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import torchvision
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('/workspace')
import os
os.chdir("/workspace")
print(os.getcwd())
from CasMVSNet_pl.models.mvsnet import CascadeMVSNet
from CasMVSNet_pl.models.mvsnet import CascadeMVSNet
from CasMVSNet_pl.utils import load_ckpt
from CasMVSNet_pl.utils import * 
from einops import rearrange
from torchvision import transforms as T
import matplotlib.pyplot as plt
from inplace_abn import ABN
import os
from debug_utils import DTUDataset, PanoOutpaintGenerator ,DTU
def decode_batch(batch):
    imgs = batch['images']
    proj_mats = batch['proj_mats']
    depths = batch['depths']
    masks = batch['mask']
    init_depth_min = batch['init_depth_min'].item()
    depth_interval = batch['depth_interval'].item()
    return imgs, proj_mats, init_depth_min, depth_interval

unpreprocess = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                           std=[1/0.229, 1/0.224, 1/0.225])
def visualize_depth(x, cmap=cv2.COLORMAP_JET):
        """
        depth: (H, W)
        """
        
        x = x.cpu().detach().numpy()
        x = np.nan_to_num(x) # change nan to 0
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
        x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
        x = (255*x).astype(np.uint8)
        x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
        x_ = T.ToTensor()(x_) # (3, H, W)
        return x_
def normalize_depth(x):
        
        x = torch.nan_to_num(x) # change nan to 0
        mi = torch.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = torch.max(x)
        x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
        x = (255*x)
       
        return x
depth_model = CascadeMVSNet(n_depths=[8,32,48],
                      interval_ratios=[1.0,2.0,4.0],
                      num_groups=1,
                      norm_act=ABN).cuda()
load_ckpt(depth_model, '/workspace/CasMVSNet_pl/ckpts/_ckpt_epoch_10.ckpt')
#depth_model.eval()
def plot_depths(batch,results,depths,path = "./"):
    
    fig, axs = plt.subplots(4, 3, figsize=(10, 10))
    gt_result,impaired_result,pred_result = results
    depth_pred =  pred_result["depth_0"]
    depth_gt =  gt_result["depth_0"]
    depth_impaired =  impaired_result["depth_0"]
    
    images,mask_images,images_pred = batch


    axs[0,0].imshow(visualize_depth(gt_result['depth_0'][0]).permute(1,2,0))
    axs[0,0].set_title("gt_depth")
    axs[0,1].imshow(visualize_depth(pred_result['depth_0'][0]).permute(1,2,0))
    axs[0,1].set_title("predicted_depth")
    axs[0,2].imshow(visualize_depth( depth_impaired[0]).permute(1,2,0))
    axs[0,2].set_title("impaired_depth")

    #unpreprocess = normalize_depth
        
    axs[1,0].imshow(unpreprocess(images[0][0]).permute(1,2,0))
    axs[1,0].set_title("image 1")
    axs[1,1].imshow(unpreprocess(images[0][1]).permute(1,2,0))
    axs[1,1].set_title("image 2")
    axs[1,2].imshow(unpreprocess(images[0][2]).permute(1,2,0))
    axs[1,2].set_title("image 3")
    axs[2,0].imshow(unpreprocess(mask_images[0][0]).permute(1,2,0).cpu().detach().numpy())
    axs[2,0].set_title("mask image 1")
    axs[2,1].imshow(unpreprocess(mask_images[0][1]).permute(1,2,0).cpu().detach().numpy())
    axs[2,1].set_title("mask image 2")
    axs[2,2].imshow(unpreprocess(mask_images[0][2]).permute(1,2,0).cpu().detach().numpy())
    axs[2,2].set_title("mask image 3")
    axs[3,0].imshow(unpreprocess(images_pred[0][0]).permute(1,2,0).cpu().detach().numpy())
    axs[3,0].set_title("predicted image 1")
    axs[3,1].imshow(unpreprocess(images_pred[0][1]).permute(1,2,0).cpu().detach().numpy())
    axs[3,1].set_title("predicted image 2")
    axs[3,2].imshow(unpreprocess(images_pred[0][2]).permute(1,2,0).cpu().detach().numpy())
    axs[3,2].set_title("predicted image 3")

    img_dir1 = os.path.join(path,"depths_prediction.png")
    img_dir2 = os.path.join(path,"observed_dephts.png")

    plt.savefig(img_dir1)
    plt.close()
    plt.imshow(visualize_depth(depths[0]).permute(1,2,0), alpha=0.5)
    
    plt.savefig(img_dir2)
    plt.close()

class PanoOutpaint(PanoOutpaintGenerator):
    def __init__(self, depth_model,plot_depths,**kwargs):
        super().__init__( **kwargs)
        self.depth_model = depth_model
        self.plot_depths = plot_depths
        self.criterion =  torch.nn.SmoothL1Loss(reduction='mean')
        self.trainable_params = [(list(self.text_encoder.parameters()),1.0)]
     
    



    def gen_cls_free_guide_pair(self, latents, timestep, prompt_embd, batch):


        R = [batch['R']]
        K = [batch['K']]

        meta = {
            'K': K,
            'R': R,
            'homographys': batch["homographys"],
            "use_corres": batch["use_corres"]
        }

        return latents, timestep, prompt_embd, meta

   
    def forward_cls_free(self, latents_high_res, _timestep, prompt_embd, batch, model):
        latents, _timestep, _prompt_embd, meta = self.gen_cls_free_guide_pair(
            latents_high_res, _timestep, prompt_embd, batch)
        #print(latents.shape)
        noise_pred = model(
            latents, _timestep, _prompt_embd, meta)

        noise_pred_text = noise_pred.chunk(2)
       

        return noise_pred
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
    def decode_latent(self,latents,vae,transformation = False):
        b, m = latents.shape[0:2]
        latents = (1 / vae.config.scaling_factor * latents)

        images = []
        #with torch.no_grad():
        for j in range(m):
            image = vae.decode(latents[:, j]).sample
            images.append(image)
        image = torch.stack(images, dim=1)

        image = (image / 2 + 0.5).clamp(0, 1) # -1 ,1 
        #image = image.permute(0, 1, 3, 4, 2).float()#.cpu().numpy()
        #image = (image * 255)#.round().astype('uint8')
        if transformation:
            transformation =  T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            image = transformation(image)
            print("transformed image shape is:",image.shape)
        return image
    
    def training_step(self,batch,batch_idx):
        depth_model=self.depth_model
        plot_depths=self.plot_depths
        images = batch["images"]
        mask_images = batch["dark_imgs"]#.cuda()
        mask = batch["mask"].float()
        bs, m, _,h, w = images.shape
        #images=rearrange(images, 'bs m h w c -> bs m c h w')
        #mask_images=rearrange(mask_images, 'bs m h w c -> bs m c h w')
        #print(images.shape,mask_images.shape)
        
        mask_latnets, masked_image_latents=self.prepare_mask_image(mask_images,batch)
        

        device = mask_images.device

        latents= torch.randn(
            bs, m, 4, h//8, w//8, device=device)

        prompt_embds = []
        for prompt in batch['prompts']:
            prompt = prompt[0]
            print(prompt)
            prompt_embds.append(self.encode_text(
                prompt, device)[0])
        prompt_embd = torch.stack(prompt_embds, dim=1)

        #prompt_null = self.encode_text('', device)[0]
        #prompt_embd = torch.cat(
        #    [prompt_null[:, None].repeat(1, m, 1, 1), prompt_embds])
        print("prompt_embd require grad is:",prompt_embd.requires_grad)
        
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
        print("latents require grad is:",latents.requires_grad)
        images_pred = self.decode_latent(
                latents, self.vae,transformation = True)
        print("images_pred require grad is:",images_pred.requires_grad)
        
        images, proj_mats, init_depth_min, depth_interval = decode_batch(batch)
        images = images.cuda()
        images.requires_grad = True
        loss = self.criterion(images_pred.float(),images.float())
        print("loss is : ",loss,loss.requires_grad,images.requires_grad)
        return loss
        gt_result = depth_model(images.cuda(), proj_mats.cuda(), init_depth_min, depth_interval)
        print(images_pred.shape,images_pred.requires_grad,images.requires_grad)
        

        impaired_result = depth_model(mask_images.cuda(), proj_mats.cuda(), init_depth_min, depth_interval)
       

        pred_result = depth_model(images_pred, proj_mats.cuda(), init_depth_min, depth_interval)
        depth_pred =  pred_result["depth_0"]
        depth_gt =  gt_result["depth_0"]
        depth_impaired =  impaired_result["depth_0"]
        
        
       
        results = gt_result,impaired_result,pred_result
        
        batch_images = images,mask_images,images_pred 
        plot_depths(batch_images,results,batch["depths"])

        return loss
    


    @torch.no_grad()
    def inference(self,batch):
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
            latents, self.vae,transformation = True)
        return images_pred



    def validation_step(self, batch, batch_idx):
        
        depth_model=self.depth_model
        plot_depths=self.plot_depths
        images_pred = self.inference(batch)
        

        images = batch["images"]
        mask_images = batch["dark_imgs"].cuda()
        depths = batch["depths"]

        
        images, proj_mats, init_depth_min, depth_interval = decode_batch(batch)

        
        gt_result = depth_model(images.cuda(), proj_mats.cuda(), init_depth_min, depth_interval)
        

        impaired_result = depth_model(mask_images.cuda(), proj_mats.cuda(), init_depth_min, depth_interval)
       

        pred_result = depth_model(images_pred, proj_mats.cuda(), init_depth_min, depth_interval)
        
        results = gt_result,impaired_result,pred_result

        try:
            img_dir = os.path.join(self.logger.log_dir, 'images')
        except:
            
            img_dir = os.path.join("./", 'validation_step_images')

        
        batch_images = images,mask_images,images_pred 
        plot_depths(batch_images,results,depths,path=img_dir)
        images = ((batch['images']/2+0.5)
                          * 255).cpu().numpy().astype(np.uint8)
        images_pred = ((images_pred.permute(0,1,3,4,2)/2+0.5)
                          * 255).cpu().numpy().astype(np.uint8)
        plt.imshow(images_pred[0][0])
        plt.show()
        
        # compute image & save
        try:
            if self.trainer.global_rank == 0:
                self.save_image(images_pred, images, batch['prompt'], batch_idx)
        except:
            print("not bound to logger and trainer")
if __name__ =="__main__":
    dataset = DTU(root_dir = "/workspace/dtu",split = "train",len = 1)
    config = yaml.load(open("/workspace/test.yaml", 'rb'), Loader=yaml.SafeLoader)
    model = PanoOutpaint(config = config,depth_model=depth_model,plot_depths=plot_depths)
    model.load_state_dict(torch.load("./weights/pano_outpaint.ckpt",
                                    map_location='cpu')['state_dict'], strict=False)
    model = model.to("cuda")
    from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.optim import Adam
train_loader = DataLoader(dataset,batch_size = 1)

optimizer = Adam(model.trainable_params[0][0], lr=1e-4)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
for i in range(200):

    for data in train_loader:
        optimizer.zero_grad()
        data["dark_imgs"] =data["dark_imgs"].cuda()
        data["imgs"] = data["imgs"].cuda()
        data["R"] = data["R"].cuda()
        data["K"]= data["K"].cuda()
        data["use_corres"]=False
        data["homographys"] = data["homographys"].float().cuda()
        #data["images"] = data["dark_imgs"].cuda()
        print(data.keys())
        
        
        #image_pred = model.inference(data)
        
        #print(image_pred.shape)

        #plt.imshow(image_pred[0][0])
        #plt.show()



        #unprocess = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
            #                                std=[1/0.229, 1/0.224, 1/0.225])
        
            
        loss = model.training_step(data,0)
        print(loss,loss.requires_grad)
        loss.backward()

        optimizer.step()
        print("now step is:",i)
        model.validation_step(data,0)