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

#### If a GPU is available its probably faster so use that. If it isnt go ahead and use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print('WARNING: Not using GPU. Consider using CUDA for possibly faster results')
#device = 'cpu'


### Set up parameters
# Directory with training images sorted as data_dir/defect and data_dir/no_defect
data_dir = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/part/sorted_training"
num_epochs = 100
batch_size = 25
learning_rate = 0.0008
# Use square images with IMSZ width and height
IMSZ = 150
do_test = True # If we want to test at the end

# If we are testing, directory for testing data folders in same format as training data
test_dir = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/paper/sorted_test"

# if we are testing, this is a single large image to cut up and test on (like what we would do to find where defects are in a total image)
test_img = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/part/part_test.png"


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
cnn = ConvolutionalNN([IMSZ, IMSZ, 1], conv_layers, np.asarray([1000,100,50]), 1, nn.RReLU)
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
lossFunc = nn.BCELoss().to(device)

#### Main training loop
losses = []
for epoch in range(num_epochs):
    end_time = time.time() # Keeping track of time if we want to show a progress bar or soemthing later
    totalCorrect = 0
    num_guess_none = 0
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
        predicted = torch.round(torch.transpose(output,1,0))
        batchCorrect = (predicted == target).sum()

        # Keep running total of correct
        num_guess_none = num_guess_none + (predicted == 1).sum()
        totalCorrect = totalCorrect + batchCorrect

    end_time = time.time()

    # Print training accuracy each epoch end
    print("Training accuracy: " + str(round(totalCorrect.cpu().numpy()/len(dataset)*10000)/100))
    losses.append(loss.detach().cpu().numpy())

##### Now test
if do_test:

    # Load testing data
    dataset = ImageFolder(data_dir,  transform = transforms.Compose([ transforms.Grayscale(), transforms.ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    totalCorrect = 0
    num_guess_none = 0
    # Iterate through testing data
    for batch_num, (img, target) in enumerate(dataloader):
        # Send data to cpu/gpu
        img = Variable(img).to(device)
        target = Variable(target).to(device)
        #print("going to run model")
        # ===================forward=====================
        output = model(img) # Get output

        # See what we got correct and keep running total
        predicted = torch.round(torch.transpose(output,1,0))
        batchCorrect = (predicted == target).sum()
        num_guess_none = num_guess_none + (predicted == 1).sum()
        totalCorrect = totalCorrect + batchCorrect

    # Clear things to free up memory
    del img
    del target
    del output
    del predicted
    # Print testing accuracy
    print("test acc:" + str(round(totalCorrect.cpu().numpy()/(batch_size*(batch_num+1))*100,2)))

    ### Now do segmneted test on a single image

    # Read in the image
    timg = io.imread(test_img, as_gray=True)
    total_size = timg.shape # Get its size

    # Initialize values that will map where the defects are in the image
    total_map = torch.zeros(total_size, dtype=torch.float16, device=device)
    map1 = torch.zeros([IMSZ,IMSZ], dtype=torch.float16, device=device)
    map2 = torch.zeros(total_size, dtype=torch.float16, device=device)

    # Initialize values for running the model
    img = torch.zeros([batch_size, 1, IMSZ, IMSZ], device=device)
    output = torch.zeros([batch_size], device=device)
    all_predictions = np.array([])

    # Load in the image using the SegmentedImage dataset (see image_segmentation.py)
    # This is where the image is cut up into smaller pecies
    # Step is how much to step between cuts (if step=IMSZ there is no overlap)
    dataset = SegmentedImage(test_img, step=75, out_size=[IMSZ, IMSZ])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=10)
    total_num_b = len(dataset) / batch_size # Total number of batches

    # Iterate through the cut up image as sub images
    for batch_num, (img, coord_x, coord_y) in enumerate(dataloader):
        # Run the model and get the output
        img = Variable(img).to(device)
        output = model(img)

        # For each one of the results, map the output as a color to the main map of where the defects are
        for i in range(len(coord_x)):
            # Get where the sub image is relative to the large image
            c = [coord_x[i], coord_y[i]]
            # Create a map of a single grayscale value representing the chance of a defect in this sub image
            map1 = torch.ones([IMSZ,IMSZ], device=device) * output[i]

            # Now put map1 into where it would be in the big image (but zeros everywhere else)
            map2 = torch.nn.functional.pad(map1, (c[1], total_size[1] - c[1] - IMSZ, c[0], total_size[0] - c[0] - IMSZ))

            # HAVE TO .detach() to keep pytorch from recording gradients ! Memory issues otherwise
            # Keep a total_map by adding together each grayscale values for all the sub images
            total_map = total_map + map2.detach()

        # Free up memory (might not still be needed bc above fix)
        del img
        del output
        del map1
        del map2
        gc.collect()

        print("Finished batch " + str(batch_num) + " of " + str(total_num_b))

    # Plot the output map of where defects might be on the large image (total_map)
    f = plt.figure(1)
    imshow(total_map.cpu().detach().numpy())
    plt.gray()
    f.show()
    plt.show()

# Below is how we could save the model if we wanted to stop part way through training or after we are done training
"""
torch.save(losses, "losses.np")
torch.save(model, "classifier.nn")
torch.save(optimizer, "optimizer.nn")
"""
