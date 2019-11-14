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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print('WARNING: Not using GPU. Consider using CUDA for possibly faster results')
#device = 'cpu'


data_dir = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects"
num_epochs = 10000
batch_size = 10
learning_rate = 0.001
IMSZ = 320

dataset = c.ImageDataset(data_dir, [IMSZ,IMSZ])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

class filter_network(nn.Module):

    def __init__(self, im_size):
        padding = 1
        super(filter_network,self).__init__()
        self.l1 = nn.Conv2d(im_size[2], 20, 3, 1, 1)
        self.l2 = nn.Conv2d(20, 40, 3, 1, 1)
        self.l3 = nn.Conv2d(40, 80, 3, 1, 1)
        self.l4 = nn.Conv2d(80, 160, 3, 1, 1)
        self.l5 = nn.Conv2d(160, 320, 3, 1, 1)
        self.l6 = nn.ConvTranspose2d(320, 160, 3, 1, 1)
        self.l7 = nn.ConvTranspose2d(160, 80, 3, 1, 1)
        self.l8 = nn.ConvTranspose2d(80, 40, 3, 1, 1)
        self.l9 = nn.ConvTranspose2d(40, 20, 3, 1, 1)
        self.l10 = nn.ConvTranspose2d(20, 10, 3, 1, 1)
        self.l11 = nn.ConvTranspose2d(10, 1, 3, 1, 1)
        nn.init.normal_(self.l1.weight)
        nn.init.normal_(self.l2.weight)
        nn.init.normal_(self.l3.weight)
        nn.init.normal_(self.l4.weight)
        nn.init.normal_(self.l5.weight)
        nn.init.normal_(self.l6.weight)
        nn.init.normal_(self.l7.weight)
        nn.init.normal_(self.l8.weight)
        nn.init.normal_(self.l9.weight)
        nn.init.normal_(self.l10.weight)
        nn.init.normal_(self.l11.weight)
        self.act_func = nn.Sigmoid()

    def forward(self, x):
        mp = nn.MaxPool2d(2)
        us = nn.Upsample(scale_factor=2)
        x = self.l1(x)
        x = self.act_func(x)
        x = mp(x)
        x = self.l2(x)
        x = self.act_func(x)
        x = mp(x)
        x = self.l3(x)
        x = self.act_func(x)
        x = mp(x)
        x = self.l4(x)
        x = self.act_func(x)
        x = mp(x)
        x = self.l5(x)
        x = self.act_func(x)
        x = mp(x)
        x = self.l6(x)
        x = self.act_func(x)
        x = us(x)
        x = self.l7(x)
        x = self.act_func(x)
        x = us(x)
        x = self.l8(x)
        x = self.act_func(x)
        x = us(x)
        x = self.l9(x)
        x = self.act_func(x)
        x = us(x)
        x = self.l10(x)
        x = self.act_func(x)
        x = us(x)
        x = self.l11(x)
        x = self.act_func(x)
        return x

    def get_features(self, x):
        x = self.l1(x)
        return x

if os.path.exists("autoencoder.nn"):
    model = torch.load("autoencoder.nn")
else:
    model = filter_network([IMSZ,IMSZ,3])

# Print number of paraemters
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params_e = sum([np.prod(p.size()) for p in model_parameters])
print('Network has ' + str(params_e))

test_img, test_defect = dataset[3]
plt.ion()
fig = plt.figure();
fim = plt.imshow(test_img.numpy().transpose(1,2,0))
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

img = next(iter(dataloader))
img = Variable(img).to(device)
output = model(img)

fim.set_data(img[0].detach().cpu().numpy().transpose(1,2,0))
fim2.set_data(output[0].detach().cpu().numpy().transpose(1,2,0))
plt.draw()
plt.pause(3600)
