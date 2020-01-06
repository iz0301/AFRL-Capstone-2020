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

class SegmentedImage(Dataset):
    def __init__(self, img_path, step=1, out_size=[150,150]):
        self.img = io.imread(img_path)
        self.in_size = self.img.shape
        self.out_size = out_size
        self.step = step

    def __len__(self):
        return math.floor(((self.in_size[0] - (self.out_size[0] - 1)) * (self.in_size[1] - (self.out_size[1] - 1))) / (self.step * self.step))

    def __getitem__(self, idx):
        img = self.img
        in_size = self.in_size
        out_size = self.out_size
        step = self.step
        # Coordinates in the origianl image where the top left pixel of the sub image falls
        coords = [math.floor(((idx * step) % (in_size[0] - (out_size[0] - 1))) / step) * step, step * math.floor((idx * step) / (in_size[0] - (out_size[0] - 1)))];

        img = TF.to_pil_image(img)
        img = TF.to_grayscale(img,num_output_channels=3)

        img = TF.crop(img, coords[0], coords[1], out_size[0], out_size[1])

        img = TF.to_tensor(img)
        return (img, coords[0], coords[1])
