import numpy as np
from skimage import io
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import time
import torch.nn as nn
from os import path
from torch.autograd import Variable
import custom_nn as c
import train as t
import torchvision.transforms.functional as TF
from custom_nn import filter_network


data_dir = "/Users/isaaczachmann/Documents/AFRL-Capstone-2020/surface-defects/Defects/modified/testing/"
IMSZ = 3024
batch_size = 10

dataset = c.ImageDataset(data_dir, [IMSZ,IMSZ])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

device = 'cpu'
test_img, test_defect = dataset[0]
mdefect = io.imread("~/Documents/AFRL-Capstone-2020/surface-defects/Defects/modified/testing/2_936.jpg")
mdefect = TF.to_pil_image(mdefect)
mdefect = TF.to_grayscale(mdefect)
mdefect = TF.to_tensor(mdefect)

plt.ion()
fig = plt.figure();
fim = plt.imshow(np.squeeze(mdefect.numpy().transpose(1,2,0)))
fig2 = plt.figure();
fim2 = plt.imshow(np.squeeze(test_defect.numpy().transpose(1,2,0)))
fig3 = plt.figure();
fim3 = plt.imshow(np.squeeze(test_defect.numpy().transpose(1,2,0)))
fim.set_cmap('Greys')
fim2.set_cmap('Greys')
fim3.set_cmap('Greys')
plt.draw()
plt.pause(1)

model = torch.load("./surface-defects/pytorch/autoencoder_l1.nn")
model = model.to(device)

for batch_num, (img, defect) in enumerate(dataloader):

    img = Variable(img).to(device)
    defect = Variable(defect).to(device)
    print("going to run model")
    # ===================forward=====================
    output = model.forward(img)
    output = output > 0.1
    print("got output")
    # ===================backward====================
    for i in range(10):
        fim.set_data(np.squeeze(img[i].detach().cpu().numpy().transpose(1,2,0)))
        fim2.set_data(np.squeeze(output[i].detach().cpu().numpy().transpose(1,2,0)))
        fim3.set_data(np.squeeze(defect[i].detach().cpu().numpy().transpose(1,2,0)))
        plt.draw()
        plt.pause(3)

print("done")
plt.pause(100)
