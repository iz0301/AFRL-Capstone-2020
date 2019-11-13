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
num_epochs = 1000
batch_size = 5
learning_rate = 0.001
IMSZ = 300

dataset = c.ImageDataset(data_dir, [IMSZ,IMSZ])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)


if os.path.exists("autoencoder.nn"):
    model = torch.load("autoencoder.nn")
else:
    model = c.Autoencoder([IMSZ,IMSZ,3],5000,[10,20,40,80,160],[180,90,50,25,15])
    model.initWeights(None)

# Print number of paraemters
model_parameters = filter(lambda p: p.requires_grad, model.encoder.parameters())
params_e = sum([np.prod(p.size()) for p in model_parameters])
model_parameters = filter(lambda p: p.requires_grad, model.decoder.parameters())
params_d = sum([np.prod(p.size()) for p in model_parameters])
print('Encoder has ' + str(params_e))
print('Decoder has ' + str(params_d))


test_img, test_defect = dataset[5]
plt.ion()
fig = plt.figure();
fim = plt.imshow(test_img.numpy().transpose(1,2,0))
fig2 = plt.figure();
fim2 = plt.imshow(np.squeeze(test_defect.numpy().transpose(1,2,0)))
fim2.set_cmap('Greys')
plt.draw()
plt.pause(1)

#model.initalize()
"""
for epoch in range(num_epochs):
    for batch_num, img in enumerate(dataloader):
        img = Variable(img).to(device)

        #print("going to run model")
        # ===================forward=====================
        output = model(img)
        #print("got output")
        loss = lossFunc(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        fim.set_data(img[0].detach().cpu().numpy().transpose(1,2,0))
        fim2.set_data(output[0].detach().cpu().numpy().transpose(1,2,0))
        plt.draw()
        plt.pause(0.0001)
"""

n_latent = 2000
best_conv = [10,20,40,80]
best_dconv = [80,40,20,10,5]
generations = 50
population = 10
lifetime = 30 #num expocs
im_sz = [150,150,3]
best_model = model
"""
for gen in range(generations):
    models = [None] * population
    convs = [None] * population
    dconvs = [None] * population
    ls = [0] * population
    for i in range(population):
        conv = best_conv + np.round(2 * np.random.normal(size=len(best_conv)))
        dconv = best_dconv + np.round(2 * np.random.normal(size=len(best_dconv)))

        bad = np.asarray(conv) < 1
        conv[bad] = 1
        bad = np.asarray(dconv) < 1
        dconv[bad] = 1

        models[i] = c.Autoencoder(im_sz,n_latent,conv,dconv)
        models[i].initWeights(best_model)
        convs[i] = conv
        dconvs[i] = dconv

    mn = 0
    for model in models:
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.00001)
        lossFunc = nn.MSELoss().to(device)

        ls[mn] = t.train(model, lossFunc, optimizer, lifetime, dataloader, device)
        print('Loss at i = ' + str(mn) + ' is ' + str(ls[mn]))

        img = next(iter(dataloader))
        img = Variable(img).to(device)
        output = model(img)


        fim.set_data(img[0].detach().cpu().numpy().transpose(1,2,0))
        fim2.set_data(output[0].detach().cpu().numpy().transpose(1,2,0))
        plt.draw()
        plt.pause(0.0001)

        mn = mn + 1
        model = model.to('cpu')

    index_min = min(range(len(ls)), key=ls.__getitem__)
    best_model = models[index_min]
    best_conv = convs[index_min]
    best_dconv = dconvs[index_min]

print(l)
"""

model = best_model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.001)
lossFunc = nn.MSELoss().to(device)

for i in range(1000):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    for mp in model_parameters:
        if np.any(np.absolute(mp.cpu().clone().detach().numpy())) > 8:
            print("PARAMETER EXPLODED!")

    finalLoss = t.train(model, lossFunc, optimizer, 1, dataloader, device)
    print(finalLoss)


    (img, truth) = next(iter(dataloader))
    img = Variable(img).to(device)
    output = model(img)

    fim.set_data(np.squeeze(truth[0].detach().cpu().numpy().transpose(1,2,0)))
    fim2.set_data(np.squeeze(output[0].detach().cpu().numpy().transpose(1,2,0)))
    plt.draw()
    plt.pause(0.0001)



torch.save(best_model, "autoencoder.nn")

img = next(iter(dataloader))
img = Variable(img).to(device)
output = model(img)

fim.set_data(img[0].detach().cpu().numpy().transpose(1,2,0))
fim2.set_data(output[0].detach().cpu().numpy().transpose(1,2,0))
plt.draw()
plt.pause(3600)
