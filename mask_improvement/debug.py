from debug_utils import DTUDataset,DTU, PanoOutpaintGenerator   
import yaml
import torch
config = yaml.load(open("./test.yaml", 'rb'), Loader=yaml.SafeLoader)
dataset = DTU(len = 1,root_dir = "./dtu",split = "train")
model = PanoOutpaintGenerator(config)
model.load_state_dict(torch.load("./weights/pano_outpaint.ckpt",
            map_location='cpu')['state_dict'], strict=False)
from torch.utils.data import DataLoader
model = model.cuda()
train_loader = DataLoader(dataset,batch_size = 1)
for data in train_loader:
    data["dark_imgs"] =data["dark_imgs"].cuda()
    #data["imgs"] = data["imgs"].cuda()
    data["R"] = data["R"].cuda()
    data["K"]= data["K"].cuda()
    data["use_corres"]=False
    data["homographys"] = data["homographys"].cuda().float()
    data["images"] = data["dark_imgs"].cuda()

    image_pred = model.inference(data)
    import matplotlib.pyplot as plt
    plt.imshow(image_pred[0][0])
    plt.savefig("test.png")
