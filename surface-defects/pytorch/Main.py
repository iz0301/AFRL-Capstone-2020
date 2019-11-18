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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print('WARNING: Not using GPU. Consider using CUDA for possibly faster results')
#device = 'cpu'


data_dir = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/modified/filtered"
num_epochs = 200
batch_size = 10
learning_rate = 0.002
IMSZ = 320

dataset = c.ImageDataset(data_dir, [IMSZ,IMSZ])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

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
fim.set_cmap('Greys')
fim2.set_cmap('Greys')
fim3.set_cmap('Greys')
plt.draw()
plt.pause(1)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.001)
lossFunc = nn.MSELoss().to(device)
losses = []
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

    end_time = time.time()
    fim.set_data(np.squeeze(img[0].detach().cpu().numpy().transpose(1,2,0)))
    fim2.set_data(np.squeeze(output[0].detach().cpu().numpy().transpose(1,2,0)))
    fim3.set_data(np.squeeze(defect[0].detach().cpu().numpy().transpose(1,2,0)))
    plt.draw()
    plt.pause(0.0001)
    print("Loss: " + str(loss))
    losses.append(loss.detach().cpu().numpy())

torch.save(losses, "losses.np")
torch.save(model, "autoencoder.nn")
torch.save(optimizer, "optimizer.nn")
"""
img = next(iter(dataloader))
img = Variable(img).to(device)
output = model(img)

fim.set_data(img[0].detach().cpu().numpy().transpose(1,2,0))
fim2.set_data(output[0].detach().cpu().numpy().transpose(1,2,0))
plt.draw()"""
plt.pause(36000)
