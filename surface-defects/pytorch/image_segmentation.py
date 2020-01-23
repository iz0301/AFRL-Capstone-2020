import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import math
from skimage import io
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import time

# Dataset I made for cutting up images
class SegmentedImage(Dataset):
    """
    img_path = path to a single large image to slice up
    step = How big of a step to take when slicing up the image
    out_size = What size should our cuts be
    """
    def __init__(self, img_path, step=1, out_size=[150,150]):
        self.img = io.imread(img_path)
        self.in_size = self.img.shape
        self.out_size = out_size
        self.step = step

    def __len__(self):
        return math.floor(((self.in_size[0] - (self.out_size[0] - 1)) * (self.in_size[1] - (self.out_size[1] - 1))) / (self.step * self.step))

    """
     Main thing is here in __getitem__ which is what is called when we do
     enumerate(DataLoader) Big thing is at the botton with what we return
     which is:
     img, x_coord, y_coord
     where img is the small section of the large image, x_coord is the
     x_coordinate relative to the large image, and y_coord is the y coordinate
     relatve to the large image. These are where the [0,0] point in this image
     occur in the big image.
    """
    def __getitem__(self, idx):
        img = self.img
        in_size = self.in_size
        out_size = self.out_size
        step = self.step
        # Coordinates in the origianl image where the top left pixel of the sub image falls
        coords = [math.floor(((idx * step) % (in_size[0] - (out_size[0] - 1))) / step) * step, step * math.floor((idx * step) / (in_size[0] - (out_size[0] - 1)))];

        img = TF.to_pil_image(img)
        img = TF.to_grayscale(img,num_output_channels=1)

        img = TF.crop(img, coords[0], coords[1], out_size[0], out_size[1])

        img = TF.to_tensor(img)
        return (img, coords[0], coords[1])
