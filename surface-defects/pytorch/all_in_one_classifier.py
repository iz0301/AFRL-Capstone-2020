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
import torchvision.transforms.functional as TF
from custom_nn import ConvolutionalNN
from torchvision.datasets import ImageFolder
import math
from image_segmentation import SegmentedImage
import gc
import sys
from show_defects import show_defects

#### If a GPU is available its probably faster so use that. If it isnt go ahead and use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print('WARNING: Not using GPU. Consider using CUDA for possibly faster results')
#device = 'cpu'


### Set up parameters
# Directory with training images sorted as data_dir/defect and data_dir/no_defect
data_dir = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/paper/flash/sorted/train"
num_epochs = 20
batch_size = 25
learning_rate = 0.0008
n_classes = 3
# Use square images with IMSZ width and height
IMSZ = 150
do_test = True # If we want to test at the end

# If we are testing, directory for testing data folders in same format as training data
test_dir = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/paper/flash/sorted/test"

# if we are testing, this is a single large image to cut up and test on (like what we would do to find where defects are in a total image)
test_img = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/paper/flash/Isaac_2_crop.png"


#### Load in the data and convert it to a grayscale tensor
dataset = ImageFolder(data_dir,  transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

### Set up our model
 # Number of channels for each convolutional layer, and length of list is how many convolutional layers
conv = [10,30,90,180]
conv_layers = np.empty([len(conv),7])
for i in range(len(conv)):
    # Set up convolutional layers and maxpool layers based on the list above and also use:
    # kernel_size=5, stride=1, padding=0, max_pool_kernel=2, max_pool_stride=2, max_pool_padding=0
    conv_layers[i] = [conv[i], 5, 1, 0, 2, 2, 0]

### Set up a pre_filter that made defects easier to see visually
pre_filter_weights = [[-5, -10, -5], [-10, 60, -10], [-5, -10, -5]];
pre_filter_weights = torch.FloatTensor(pre_filter_weights)
pre_filter_weights = pre_filter_weights.repeat(1, 1, 1, 1)
pre_filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
pre_filter.weight.data = pre_filter_weights
pre_filter.weight.requires_grad = True # Let the filter weights change during training

# Create the convolutional nural network using the class in custom_nn.py
# Use 3 fully connected layers and 1 output layer. Fully connected connected
# layers have 1000, 100, then 50 nodes. Output layer has 1 node
# Use random rectified linear unit as the activation function (see doc)
cnn = ConvolutionalNN([IMSZ, IMSZ, 1], conv_layers, np.asarray([1000,100,50]), n_classes, nn.RReLU)
cnn.init_weights(nn.init.calculate_gain) # Initialize wieghts
 # Add the pre filter + the convolutional nural network + sigmoid activation function to get the full model
 # Sigmoid activation makes output between -1 and 1
 # (maybe we should change labels to be -1 and 1 then)
model = nn.Sequential(pre_filter, cnn, nn.Sigmoid())



# Print number of trainable model paraemters for debuging purposes
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params_e = sum([np.prod(p.size()) for p in model_parameters])
print('Network has ' + str(params_e))

# Put the model on the gpu/cpu
model = model.to(device)
# Using the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.00001)

# Using binary cross entropy loss function
lossFunc = nn.CrossEntropyLoss().to(device)

#### Main training loop
losses = []
for epoch in range(num_epochs):
    end_time = time.time() # Keeping track of time if we want to show a progress bar or soemthing later
    totalCorrect = 0
    # loop through the data
    for batch_num, (img, target) in enumerate(dataloader):
        start_time = time.time()

        # Put the data onto gpu/cpu
        img = Variable(img).to(device)
        target = Variable(target).to(device)
        # ===================forward=====================
        output = model(img) # Get output
        loss = lossFunc(output, target.float()) # calculate loss
        # ===================backward====================
        optimizer.zero_grad() # Clear old gradients
        loss.backward() # Calculate new gradients
        optimizer.step() # Update weights based on gradients

        # Calculate how many were predicted correctly
        #predicted = torch.round(torch.transpose(output,1,0))
        _, predicted = torch.max(output, 1)
        batchCorrect = (predicted == target).sum()

        # Keep running total of correct
        totalCorrect = totalCorrect + batchCorrect

    end_time = time.time()

    # Print training accuracy each epoch end
    print("Training accuracy: " + str(round(totalCorrect.cpu().numpy()/len(dataset)*10000)/100))
    losses.append(loss.detach().cpu().numpy())
