import os
import yaml
import pickle
import argparse
import importlib
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import torchvision
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('/input0')
import os
os.chdir("..")
from CasMVSNet_pl.models.mvsnet import CascadeMVSNet


from torchvision import transforms as T

from inplace_abn import ABN
import os
from debug_utils import DTUDataset, PanoOutpaintGenerator ,DTU
def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/DTU/mvs_training/dtu/',
                        help='root directory of dtu dataset')
    parser.add_argument('--dataset_name', type=str, default='dtu',
                        choices=['dtu', 'tanks', 'blendedmvs'],
                        help='which dataset to train/val')
    parser.add_argument('--split', type=str, default='test',
                        help='which split to evaluate')
    parser.add_argument('--scan', type=str, default='',
                        help='specify scan to evaluate (must be in the split)')
    parser.add_argument('--cpu', default=False, action='store_true',
                        help='''use cpu to do depth inference.
                                WARNING: It is going to be EXTREMELY SLOW!
                                about 37s/view, so in total 30min/scan. 
                             ''')
    # for depth prediction
    parser.add_argument('--n_views', type=int, default=5,
                        help='number of views (including ref) to be used in testing')
    parser.add_argument('--depth_interval', type=float, default=2.65,
                        help='depth interval unit in mm')
    parser.add_argument('--n_depths', nargs='+', type=int, default=[8,32,48],
                        help='number of depths in each level')
    parser.add_argument('--interval_ratios', nargs='+', type=float, default=[1.0,2.0,4.0],
                        help='depth interval ratio to multiply with --depth_interval in each level')
    parser.add_argument('--num_groups', type=int, default=1, choices=[1, 2, 4, 8],
                        help='number of groups in groupwise correlation, must be a divisor of 8')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[1152, 864],
                        help='resolution (img_w, img_h) of the image, must be multiples of 32')
    parser.add_argument('--ckpt_path', type=str, default='ckpts/exp2/_ckpt_epoch_10.ckpt',
                        help='pretrained checkpoint path to load')
    parser.add_argument('--save_visual', default=False, action='store_true',
                        help='save depth and proba visualization or not')

    # for point cloud fusion
    parser.add_argument('--conf', type=float, default=0.999,
                        help='min confidence for pixel to be valid')
    parser.add_argument('--min_geo_consistent', type=int, default=5,
                        help='min number of consistent views for pixel to be valid')
    parser.add_argument('--max_ref_views', type=int, default=400,
                        help='max number of ref views (to limit RAM usage)')
    parser.add_argument('--skip', type=int, default=1,
                        help='''how many points to skip when creating the point cloud.
                                Larger = fewer points and smaller file size.
                                Ref: skip=10 creates ~= 3M points = 50MB file
                                     skip=1 creates ~= 30M points = 500MB file
                             ''')

    return parser.parse_args()

sys.path.append('Modules/biHomE')

print(os.getcwd())
import src

from Modules.biHomE.src.data import transforms as transform_module
from Modules.biHomE.src.utils.checkpoint import CheckPointer
def decode_batch(batch):
    imgs = batch['imgs']
    
    proj_mats = batch['proj_mats']
    init_depth_min = batch['init_depth_min'].item()
    depth_interval = batch['depth_interval'].item()
    scan, vid = batch['scan_vid']
    return imgs, proj_mats, init_depth_min, depth_interval
class ModelWrapper(torch.nn.Sequential):
    def __init__(self, *args):
        super(ModelWrapper, self).__init__(*args)

    def predict_homography(self, data):
        for idx, m in enumerate(self):
            data = m.predict_homography(data)
        return data


if __name__ == "__main__":
    ###############################################
    args =  get_opts()
    depth_model = CascadeMVSNet(n_depths=args.n_depths,
                          interval_ratios=args.interval_ratios,
                          num_groups=args.num_groups,
                          norm_act=ABN)
    
    #########################################################
    
    
    
    dataset = DTU(root_dir = "./dtu",split = "train")
    config = yaml.load(open("./test.yaml", 'rb'), Loader=yaml.SafeLoader)
    dataset = DTU(root_dir = "./dtu",split = "train")
    print("load model2.......")
    #model = PanoOutpaintGenerator(config)
    #model.load_state_dict(torch.load("./weights/pano_outpaint.ckpt",
    #                map_location='cpu')['state_dict'], strict=False)
    from torch.utils.data import DataLoader
    #model = model.cuda()
    train_loader = DataLoader(dataset,batch_size = 1)
    for data in train_loader:
        #data["dark_imgs"] =data["dark_imgs"].cuda()
        #data["imgs"] = data["imgs"].cuda()
        #data["R"] = data["R"].cuda()
        #data["K"]= data["K"].cuda()
        #data["use_corres"]=False
        #data["homographys"] = data["homographys"].cuda().float()
        #data["images"] = data["dark_imgs"].cuda()
        print(data.keys())
        #unprocess = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
        #                                std=[1/0.229, 1/0.224, 1/0.225])
        
        imgs, proj_mats, init_depth_min, depth_interval = decode_batch(data)
        #imgs = unprocess(imgs)
        print(imgs.shape)
        result = depth_model(imgs, proj_mats, init_depth_min, depth_interval)
        import matplotlib.pyplot as plt
        plt.imshow(result["depth_0"][0].detach().numpy())
        #image_pred = model.inference(data)
        #import matplotlib.pyplot as plt
        # plt.imshow(image_pred[0][0])
        # plt.savefig("test.png")

