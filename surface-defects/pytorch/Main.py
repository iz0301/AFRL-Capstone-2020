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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print('WARNING: Not using GPU. Consider using CUDA for possibly faster results')
#device = 'cpu'


data_dir = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/modified/filtered"
num_epochs = 10000
batch_size = 10
learning_rate = 0.001
IMSZ = 320

dataset = c.ImageDataset(data_dir, [IMSZ,IMSZ])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

class filter_network(nn.Module):

    def __init__(self, im_size):
        super(filter_network,self).__init__()
        self.f1 = nn.Conv2d(im_size[2], 1, 3, 1, 1)
        self.f2 = nn.ConvTranspose2d(30, 1, 3, 1, 1)

        self.e1 = nn.Conv2d(im_size[2], 10, 3, 1, 1)
        self.e2 = nn.Conv2d(10, 20, 3, 1, 1)
        self.e3 = nn.Conv2d(20, 40, 3, 1, 1)
        self.e4 = nn.Conv2d(40, 80, 3, 1, 1)
        self.e5 = nn.Conv2d(80, 160, 3, 1, 1)
        self.e6 = nn.Conv2d(160, 320, 3, 1, 1)
        self.e7 = nn.Conv2d(320, 640, 3, 1, 1)
        self.e8 = nn.Conv2d(640, 1280, 3, 1, 1)
        self.d1 = nn.ConvTranspose2d(1280, 640, 3, 1, 1)
        self.d2 = nn.ConvTranspose2d(640, 320, 3, 1, 1)
        self.d3 = nn.ConvTranspose2d(320, 160, 3, 1, 1)
        self.d4 = nn.ConvTranspose2d(160, 80, 3, 1, 0)
        self.d5 = nn.ConvTranspose2d(80, 40, 3, 1, 1)
        self.d6 = nn.ConvTranspose2d(40, 20, 3, 1, 1)
        self.d7 = nn.ConvTranspose2d(20, 10, 3, 1, 1)
        self.d8 = nn.ConvTranspose2d(10, 5, 3, 1, 1)
        self.d9 = nn.ConvTranspose2d(5, 1, 3, 1, 1)

        nn.init.normal_(self.e1.weight)
        nn.init.normal_(self.e2.weight)
        nn.init.normal_(self.e3.weight)
        nn.init.normal_(self.e4.weight)
        nn.init.normal_(self.e5.weight)
        nn.init.normal_(self.e6.weight)
        nn.init.normal_(self.e7.weight)
        nn.init.normal_(self.e8.weight)
        nn.init.normal_(self.d1.weight)
        nn.init.normal_(self.d2.weight)
        nn.init.normal_(self.d3.weight)
        nn.init.normal_(self.d4.weight)
        nn.init.normal_(self.d5.weight)
        nn.init.normal_(self.d6.weight)
        nn.init.normal_(self.d7.weight)
        nn.init.normal_(self.d8.weight)
        nn.init.normal_(self.d9.weight)

        self.mp = nn.MaxPool2d(2)
        self.us = nn.Upsample(scale_factor=2)
        self.act_func = nn.Tanh()
        #self.encoder = nn.Sequential(\
        #self.e1, self.act_func, self.mp, self.e2, self.act_func, self.mp, self.e3, self.act_func, self.mp, self.e4, self.act_func, self.mp, self.e5, self.act_func, self.mp, self.e6, self.act_func, self.mp, self.e7, self.act_func, self.mp, self.e8, self.act_func, self.mp)
        #self.decoder = nn.Sequential(\
        #self.d1, self.act_func, self.us, self.d2, self.act_func, self.us, self.d3, self.act_func, self.us, self.d4, self.act_func, self.us, self.d5, self.act_func, self.us, self.d6, self.act_func, self.us, self.d7, self.act_func, self.us, self.d8, self.act_func, self.us, self.d9)
        self.encoder = nn.Sequential(\
        self.e1, self.act_func, self.mp, self.e2, self.act_func, self.mp)
        self.decoder = nn.Sequential(\
        self.d7, self.act_func, self.us, self.d8, self.act_func, self.us, self.d9, self.act_func)


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        #x = self.f1(x)
        return x

    def get_features(self, x):
        x = self.l1(x)
        return x

if os.path.exists("autoencoder.nn"):
    model = torch.load("autoencoder.nn")
else:
    model = filter_network([IMSZ,IMSZ,1])

# Print number of paraemters
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params_e = sum([np.prod(p.size()) for p in model_parameters])
print('Network has ' + str(params_e))

test_img, test_defect = dataset[0]
mdefect = io.imread("/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/modified/0_0_defects.jpg")
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
fim2.set_cmap('Greys')
plt.draw()
plt.pause(1)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.001)
lossFunc = nn.MSELoss().to(device)

for epoch in range(num_epochs):
    end_time = time.time()
    for batch_num, (img, defect) in enumerate(dataloader):

        start_time = time.time()
        #print("Between batches took " + str(time.time() - end_time))
        img = Variable(img).to(device)
        defect = Variable(defect).to(device)
        #print("going to run model")
        # ===================forward=====================
        output = model(img)
        #print("got output")
        loss = lossFunc(output, defect)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Loss: " + str(loss))
        end_time = time.time()
        fim.set_data(np.squeeze(img[0].detach().cpu().numpy().transpose(1,2,0)))
        fim2.set_data(np.squeeze(output[0].detach().cpu().numpy().transpose(1,2,0)))
        fim3.set_data(np.squeeze(defect[0].detach().cpu().numpy().transpose(1,2,0)))
        plt.draw()
        plt.pause(0.0001)

torch.save(model, "autoencoder.nn")
"""
img = next(iter(dataloader))
img = Variable(img).to(device)
output = model(img)

fim.set_data(img[0].detach().cpu().numpy().transpose(1,2,0))
fim2.set_data(output[0].detach().cpu().numpy().transpose(1,2,0))
plt.draw()"""
plt.pause(36000)
