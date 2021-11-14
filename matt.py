import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from inference_utils import VideoReader, VideoWriter
from skimage import io as imageio
from tqdm import tqdm
from model import MattingNetwork
import os
import numpy as np
from glob import glob

model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
model.load_state_dict(torch.load('./pretrained/rvm_mobilenetv3.pth'))

datadirs = glob('../ViewConsistencyEval/giraffe/seed*')
datadirs.sort()

for datadir in tqdm(datadirs):
    image_list = glob(os.path.join(datadir, 'images_raw/*.png'))
    image_list = sorted(image_list)

    bgr = torch.tensor([0., 0., 0.]).view(3, 1, 1).cuda()  # Black background.
    rec = [None] * 4                                       # Initial recurrent states.
    downsample_ratio = 0.25                                # Adjust based on your video.

    outdir = os.path.join(datadir, 'images')
    os.makedirs(outdir, exist_ok=True)

    with torch.no_grad():
        for i, image_path in enumerate(image_list):
            img = imageio.imread(image_path)
            img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda()
            fgr, pha, *rec = model(img, *rec, downsample_ratio)  # Cycle the recurrent states.
            com = fgr * pha + bgr * (1 - pha)              # Composite to green background.
            imageio.imsave(os.path.join(outdir, '{:0>3d}.png'.format(i)), (com.squeeze().permute(1, 2, 0) * 255.0).cpu().numpy())
