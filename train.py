
import os


import torch
import numpy as np
import argparse
from src.dataset import MP3Ddataset, Scannetdataset,ScannetDepthdataset,DTUDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from src.lightning_pano_gen import PanoGenerator
from src.lightning_depth import DepthGenerator
from src.lightning_pano_outpaint import PanoOutpaintGenerator
from src.lightning_image_depth import Images2DepthGenerator
from src.lightning_depth_dreamer import DepthDreamerGenerator
def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=0)
    parser.add_argument(
        '--ckpt_path', type=str, default= None,#"depth.ckpt",
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    torch.set_float32_matmul_precision('medium')
    config = yaml.load(open(args.main_cfg_path, 'rb'), Loader=yaml.SafeLoader)
    config['train']['max_epochs'] = args.max_epochs
    config['train']['batch_size'] = args.batch_size

    if config['dataset']['name'] == 'mp3d':
        train_dataset = MP3Ddataset(
            config['dataset'], mode='train')
        val_dataset = MP3Ddataset(
            config['dataset'], mode='val')
    elif config['dataset']['name'] == 'scannet':
        train_dataset = Scannetdataset(config['dataset'], mode='train')
        val_dataset = Scannetdataset(config['dataset'], mode='val')
    elif config['dataset']['name'] == 'scannetdepth':
        train_dataset = Scannetdataset(config['dataset'], mode='train')
        val_dataset = Scannetdataset(config['dataset'], mode='val')
    elif config['dataset']['name'] == 'dtu':
        train_dataset = DTUDataset('dtu', 'train',
                                   n_views=5, depth_interval=2.65,
                                   img_wh=None)
        val_dataset = DTUDataset('dtu', 'test',
                                 n_views=5, depth_interval=2.65,
                                 img_wh=None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)

    if config['model']['model_type'] == 'pano_generation':
        model = PanoGenerator(config)
    elif config['model']['model_type'] == 'depth':
        model = DepthGenerator(config)
    elif config['model']['model_type'] == 'pano_generation_outpaint':
        model = PanoOutpaintGenerator(config)
    elif config['model']['model_type'] == 'image_depth_generation':
        model = Images2DepthGenerator(config)
    elif config['model']['model_type'] == 'image_dreamer':
        model = DepthDreamerGenerator(config)
    elif config['model']['model_type'] == 'dtu':
        model = PanoOutpaintGenerator(config)


    if args.ckpt_path is not None:
        model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu')['state_dict'], strict=False)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="train_loss",
                                          mode="min", save_last=1,
                                          filename='epoch={epoch}-loss={train_loss:.4f}')

    # test
    ''' 
    for batch in train_loader:
        model.cuda()
        index = batch["index"]
        if f'{index}.npy' in os.listdir():
            continue
        image_pred = model.inference(batch,prompt_change = False)
        
       
        
        images = image_pred
        for i,imgs in enumerate(images):
            imgs = imgs
            index = batch["index"][i]
            np.save(f'{index}.npy', imgs)
            print(f'{index}.npy')

        
        print(images.shape)
    '''

    logger = TensorBoardLogger(
        save_dir='openbayes/home/tf_dir', name=args.exp_name, default_hp_metric=False)


    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        logger=logger)
    trainer.fit(model, train_loader, train_loader)

