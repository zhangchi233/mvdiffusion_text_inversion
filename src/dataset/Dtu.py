from torch.utils.data import Dataset
try:
    from .utils import read_pfm
except:
    from utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import time
class DTUDataset(Dataset):
    def __init__(self, root_dir, split, n_views=3, levels=3, depth_interval=2.65,
                 img_wh=None):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'val', 'test'], \
            'split must be either "train", "val" or "test"!'
        self.img_wh = img_wh
        if img_wh is not None:
            assert img_wh[0]%32==0 and img_wh[1]%32==0, \
                'img_wh must both be multiples of 32!'
        self.build_metas()
        self.n_views = n_views
        self.levels = levels # FPN levels
        self.depth_interval = depth_interval
        self.build_proj_mats()
        self.define_transforms()
        #self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        #self.clip2 = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        #self.clip2.to(device)

    def build_metas(self):
        self.metas = []
        with open(f'/input0/CasMVSNet_pl/datasets/lists/dtu/{self.split}.txt') as f:
            self.scans = [line.rstrip() for line in f.readlines()][:1]


        # light conditions 0-6 for training
        # light condition 3 for testing (the brightest?)
        light_idxs = [3] if self.img_wh else range(7)
        dark_indexes = [5] if self.img_wh else range(7)

        pair_file = "Cameras/pair.txt"
        for scan in self.scans:
            if os.path.exists(f"/input0/dtu/Depths/{scan}"):

                with open(os.path.join(self.root_dir, pair_file)) as f:
                    num_viewpoint = int(f.readline())
                    # viewpoints (49)
                    for _ in range(num_viewpoint):
                        ref_view = int(f.readline().rstrip())
                        src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                        for light_idx in light_idxs:
                            self.metas += [(scan, light_idx, ref_view, src_views)]

    def build_proj_mats(self):
        proj_mats = []
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
            for l in reversed(range(self.levels)):
                proj_mat_l = np.eye(4)
                proj_mat_l[:3, :4] = intrinsics @ extrinsics[:3, :4]
                intrinsics[:2] *= 2 # 1/4->1/2->1
                proj_mat_ls += [torch.FloatTensor(proj_mat_l)]
            # (self.levels, 4, 4) from fine to coarse
            proj_mat_ls = torch.stack(proj_mat_ls[::-1])

            proj_mats += [(proj_mat_ls, depth_min,intrinsics,extrinsics)]

        self.proj_mats = proj_mats

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        return intrinsics, extrinsics, depth_min

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32) # (1200, 1600)
        #depth = cv2.resize(depth,(1200,1600),
        #                         interpolation=cv2.INTER_NEAREST)
        if self.img_wh is None:
            depth = cv2.resize(depth, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST) # (600, 800)
            depth_0 = depth[44:556, 80:720] # (512, 640)
        else:
            depth_0 = cv2.resize(depth, self.img_wh,
                                 interpolation=cv2.INTER_NEAREST)
        depth_1 = cv2.resize(depth_0, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)
        depth_2 = cv2.resize(depth_1, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)

        depths = {"level_0": torch.FloatTensor(depth_0),
                  "level_1": torch.FloatTensor(depth_1),
                  "level_2": torch.FloatTensor(depth_2)}

        return depths

    def read_mask(self, filename):
        mask = cv2.imread(filename, 0) # (1200, 1600)
        #mask = cv2.resize(mask, (1200, 1600),
        #                        interpolation=cv2.INTER_NEAREST)
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

        masks = {"level_0": torch.BoolTensor(mask_0),
                 "level_1": torch.BoolTensor(mask_1),
                 "level_2": torch.BoolTensor(mask_2)}

        return masks

    def define_transforms(self):
        if self.split == 'train': # you can add augmentation here
            self.transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                                       ])
        else:
            self.transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                                       ])

    def __len__(self):
        return 1   #len(self.metas)

    def __getitem__(self, idx):
        sample = {}
        scan, light_idx, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs = []
        proj_mats = [] # record proj mats between views
        Ks = []
        Rs = []
        prompts = []
        dark_imgs = []
        blur_imgs = []
        depths = []
        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            if self.img_wh is None:
                img_filename = os.path.join(self.root_dir,
                                f'Rectified/{scan}_train/rect_{vid+1:03d}_{light_idx}_r5000.png')
                dark_img_filename = os.path.join(self.root_dir,
                                f'Rectified/{scan}_train/rect_{vid+1:03d}_{0}_r5000.png')
                mask_filename = os.path.join(self.root_dir,
                                f'Depths/{scan}/depth_visual_{vid:04d}.png')
                
                prompt_filename = os.path.join(self.root_dir,
                                               f'prompts/{scan}_train_prompt/rect_{vid+1:03d}_{light_idx}_r5000.png.txt')
                
                depth_filename = os.path.join(self.root_dir,
                                f'Depths/{scan}/depth_map_{vid:04d}.pfm')
            else:
                img_filename = os.path.join(self.root_dir,
                                f'Rectified/{scan}_train/rect_{vid+1:03d}_{light_idx}_r5000.png')

            img = Image.open(img_filename)
            dark_img = Image.open(dark_img_filename)

            blur_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            blur_img = cv2.GaussianBlur(blur_img,(0,0),10)
            _, threhold = cv2.threshold(blur_img, 200, 255, cv2.THRESH_BINARY)
            shadow_img = cv2.addWeighted(np.array(img), 0.3,cv2.cvtColor(threhold,cv2.COLOR_GRAY2RGB), 0.7, 0)

            try:
                with open(prompt_filename,"r") as f:
                    import re
                    generated_text = f.readline()
                    new_sentence = re.sub(r'(a.*?a)\s\w+', r'\1 rabbit', generated_text, count=1)
                    generated_text =new_sentence.replace("a rabbit", "a {0} rabbit".format("*"))
                    
            except:
                generated_text= ""
            prompts.append(generated_text)
            final = time.time()
            #img = img.resize((1200,1600), Image.BILINEAR)
            if self.img_wh is not None:
                img = img.resize(self.img_wh, Image.BILINEAR)
                dark_img = dark_img.resize(self.img_wh, Image.BILINEAR)
            #img = self.transform(img)
            imgs += [T.ToTensor()(img)]
            dark_imgs += [T.ToTensor()(dark_img)]
            blur_imgs += [T.ToTensor()(shadow_img)]
            proj_mat_ls, depth_min,intrinsics,extrinsics = self.proj_mats[vid]
            depths += [self.read_depth(depth_filename)["level_0"]]
            if i == 0:  # reference view
                sample['init_depth_min'] = torch.FloatTensor([depth_min])
                if self.img_wh is None:

                    sample['masks'] = self.read_mask(mask_filename)

                ref_proj_inv = torch.inverse(proj_mat_ls)

            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]
            Ks.append(torch.tensor(intrinsics))
            Rs.append(torch.tensor(extrinsics))
        Ks = torch.stack(Ks)
        Rs = torch.stack(Rs)
        imgs = torch.stack(imgs) # (V, 3, H, W)
        depths = torch.stack(depths)
        # blur the images


        dark_imgs = torch.stack(dark_imgs)
        proj_mats = torch.stack(proj_mats)[:,:,:3] # (V-1, self.levels, 3, 4) from fine to coarse

        sample['images'] = imgs.permute(0,2,3,1).float()#.to(self.device)
        sample['proj_mats'] = proj_mats.float()#.to(self.device)
        sample['depth_interval'] = torch.FloatTensor([self.depth_interval])#.to(self.device)
        sample['scan_vid'] = (scan, ref_view)
        sample["K"] = Ks.float()#.to(self.device)
        sample["poses"] = Rs.float()#.to(self.device)
        sample["prompt"] =prompts
    
        sample["index"] = idx
        sample["mask"] = torch.ones(sample["images"].shape[0])
        sample["depths"] = depths
        sample["dark_images"] = dark_imgs.permute(0,2,3,1).float()#.to(self.device)
        sample["blur_images"] = torch.stack(blur_imgs).permute(0,2,3,1).float()#.to(self.device)

        depth_valid_mask = depths > 0
        depth_inv = 1. / (depths + 1e-6)
        depth_max = [depth_inv[i][depth_valid_mask[i]].max()
                     for i in range(depth_inv.shape[0])]
        depth_min = [depth_inv[i][depth_valid_mask[i]].min()
                     for i in range(depth_inv.shape[0])]
        depth_max = np.stack(depth_max, axis=0)[:, None, None]
        depth_min = np.stack(depth_min, axis=0)[
            :, None, None]  # [num_views, 1, 1]
        depth_inv_norm_full = (depth_inv - depth_min) / \
            (depth_max - depth_min + 1e-6) * 2 - 1  # [-1, 1]
        depth_inv_norm_full[~depth_valid_mask] = -2
        depth_inv_norm_full = depth_inv_norm_full.to(torch.float32)
        sample["depth_inv_norm_full"] = torch.FloatTensor(depth_inv_norm_full)
        sample["depth_inv_norm_small"] = np.stack([cv2.resize(depth_inv_norm_full[i].cpu().detach().numpy(), (
            depth_inv_norm_full.shape[2]//8, depth_inv_norm_full.shape[1]//8), interpolation=cv2.INTER_NEAREST) for i in range(depth_inv_norm_full.shape[0])])


        x_pr = np.load(f"model_pr/{idx}.npy")
        x_pr = [T.ToTensor()(img) for img in x_pr]
        x_pr = torch.stack(x_pr)

        sample["x_pr"] = x_pr.permute(0,2,3,1)

        return sample
if __name__ =="__main__":

    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    data = DTUDataset("dtu",
                      "train", n_views=5, levels=3, depth_interval=2.65,)
    import matplotlib.pyplot as plt
    plt.imshow(data[0]["x_pr"][0])
    plt.savefig("test.png")