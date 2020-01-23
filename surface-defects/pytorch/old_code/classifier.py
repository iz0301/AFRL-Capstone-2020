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
from custom_nn import ConvolutionalNN
from torchvision.datasets import ImageFolder
import math
from image_segmentation import SegmentedImage


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print('WARNING: Not using GPU. Consider using CUDA for possibly faster results')
#device = 'cpu'


data_dir = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/sorted"
#data_dir = ""/home/isaac/Python/pytorch/datasets/mnist_png/testing"
num_epochs = 100
batch_size = 50
learning_rate = 0.0008
#IMSZ = 28
IMSZ = 150
do_test = True

dataset = ImageFolder(data_dir,  transform = transforms.Compose([transforms.ToTensor()]))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)


conv = [10,30,90,180]
conv_layers = np.empty([len(conv),7])
for i in range(len(conv)):
    conv_layers[i] = [conv[i], 5, 1, 0, 2, 2, 0]

pre_filter_weights = [[-5, -10, -5], [-10, 60, -10], [-5, -10, -5]];
pre_filter_weights = torch.FloatTensor(pre_filter_weights)
pre_filter_weights = pre_filter_weights.repeat(3, 3, 1, 1)
pre_filter = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, bias=False)
pre_filter.weight.data = pre_filter_weights
pre_filter.weight.requires_grad = False

cnn = ConvolutionalNN([IMSZ, IMSZ, 3], conv_layers, np.asarray([1000,100,50]), 2, nn.RReLU)
cnn.init_weights(nn.init.calculate_gain)
model = nn.Sequential(pre_filter, cnn, nn.Softmax())


# Print number of paraemters
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params_e = sum([np.prod(p.size()) for p in model_parameters])
print('Network has ' + str(params_e))

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.00001)

# CHANGE weights based on data skew [399,601]
num_no_defect = 387;
num_defect = 588;
lossFunc = nn.CrossEntropyLoss(weight=torch.FloatTensor([num_no_defect, num_defect])).to(device)

losses = []
for epoch in range(num_epochs):
    end_time = time.time()
    totalCorrect = 0
    num_guess_none = 0
    for batch_num, (img, target) in enumerate(dataloader):

        start_time = time.time()
        #print("Between batches took " + str(time.time() - end_time))
        img = Variable(img).to(device)
        target = Variable(target).to(device)
        #print("going to run model")
        # ===================forward=====================
        output = model(img)
        #print("got output")

        loss = lossFunc(output, target)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        batchCorrect = (predicted == target).sum()
        num_guess_none = num_guess_none + (predicted == 1).sum()
        totalCorrect = totalCorrect + batchCorrect
        #print(predicted)
        #print(target)
    end_time = time.time()

    print("Training accuracy: " + str(round(totalCorrect.cpu().numpy()/(batch_size*(batch_num+1))*100,2)))
    losses.append(loss.detach().cpu().numpy())

# Now test:
if do_test:
    test_dir = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/sorted_test"

    dataset = ImageFolder(data_dir,  transform = transforms.Compose([transforms.ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    totalCorrect = 0
    num_guess_none = 0
    for batch_num, (img, target) in enumerate(dataloader):
        img = Variable(img).to(device)
        target = Variable(target).to(device)
        #print("going to run model")
        # ===================forward=====================
        output = model(img)

        _, predicted = torch.max(output.data, 1)
        batchCorrect = (predicted == target).sum()
        num_guess_none = num_guess_none + (predicted == 1).sum()
        totalCorrect = totalCorrect + batchCorrect
    print("test acc:" + str(round(totalCorrect.cpu().numpy()/(batch_size*(batch_num+1))*100,2)))

# Now do segmneted test:
    test_img = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/IMG_1278.jpg"
    timg = io.imread(test_img, as_gray=True)
    total_size = timg.shape
    total_map = torch.zeros(total_size)

    map = torch.zeros([IMSZ,IMSZ])
    map2 = torch.zeros(total_size)

    map = Variable(map).to(device)
    map2 = Variable(map2).to(device)
    total_map = Variable(total_map).to(device)
    all_predictions = np.array([])

    dataset = SegmentedImage(test_img, step=150, out_size=[IMSZ, IMSZ])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=5)
    total_num_b = len(dataset) / batch_size
    for batch_num, (img, coord_x, coord_y) in enumerate(dataloader):
        img = Variable(img).to(device)
        output = model(img)
        _, predicted = torch.max(output.data, 1)

        predicted[predicted == 0] = -1
        #predicted = predicted.cpu().numpy()

        #all_predictions = np.append(all_predictions, predicted.cpu().numpy())
        # This loop below is slow
        #print("Looping...")
        start_time = time.time()
        for i in range(len(coord_x)):
            c = [coord_x[i], coord_y[i]]
            map = torch.ones([IMSZ,IMSZ], device=device) * predicted[i]
            #print("coord: " + str(c[0]) + ", " + str(c[1]))
            # These two are the slowest:
            map2 = torch.nn.functional.pad(map, (c[1], total_size[1] - c[1] - IMSZ, c[0], total_size[0] - c[0] - IMSZ))
            total_map = total_map + map2
        #print("show")
        #imshow(total_map)
        #plt.gray()
        #print("shd")
        #plt.show()
        #print("done")
        #print("Total loop time: " + str(time.time() - start_time))

        # Try without a loop:
        #predicted = np.transpose(np.atleast_3d(predicted), (2,1,0))
        #map = np.ones([IMSZ,IMSZ,batch_size]) * predicted
        #padded_map = np.zeros(total_size)
        #padded_map[]
        print("Finished batch " + str(batch_num) + " of " + str(total_num_b))
    print("show")
    f = plt.figure(1)
    imshow(total_map)
    plt.gray()
    print("shd")
    f.show()

    """
    plt.figure(2)
    for i in range(len(dataset)):
        imshow(np.swapaxes(dataset[i][0].numpy(), 0, 2))
        print("Predicted: " + str(all_predictions[i]))
        plt.gray()
        plt.show()
        """
    plt.show()
    print("done")
"""
torch.save(losses, "losses.np")
torch.save(model, "classifier.nn")
torch.save(optimizer, "optimizer.nn")
"""
