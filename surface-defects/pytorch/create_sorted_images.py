import torchvision
import torch
import os
import numpy as np
import math
from skimage import io
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import time
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

#### This file creates sorted images from 2 images (a part image and a defect map)
# These transforms could all be done while training, however they take a long
# time so doing them before hand allows more model types to be tried faster with
# The same datasets
# Updating to sort based on type of defect as well

# How many images to generate and what size should they be
num_imgs = 100
img_size = [150, 150]
# Defects are looked for in this order so last should be no defect
# 4th number is 1 for alpha values
defect_colors = [[1,0,0,1],[0,1,0,1],[1,1,1,1]]
defect_names = ["pencil","eraser","none"]
no_defect_index = 2

# prefix in case you generate images multiple times they wont overwrite eachother if they have different prefix
prefix = "i2"

# Path pointing to the folder that contains the large image and the defect map of the image
# The defect map should be pure black where the defects are and should be named IMG_NAME_defects.png
path = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/paper/flash/test_imgs/"
img = "Isaac_2_crop" # Without extension

# Where to save the output images
out_dir = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/paper/flash/sorted/binary/testing/"

# Load the original image and the defect map
og_img = io.imread(path + os.path.sep + img + ".png")
og_defect = io.imread(path + os.path.sep + img + "_defects.png")

# Make directories to save the output images
for dn in defect_names:
    os.makedirs(out_dir + os.sep + dn, exist_ok=True)

# Running total of how many of each type of defect
num_defects = [0] * len(defect_names)

# Loop to make all the images
for i in range(num_imgs): # I keeps track of image number so we can save them with unique names
    # num_ries is how many times to loop through, looking for a defect, before we
    # give up and just save a no-defect image. This should change relative to
    # the density of defects to get roughly a 50/50 split of defect and no-defect
    num_tries = 2
    # tries is the running counter of how many times we tried
    tries = 0

    defect_type = -1 # Keep track of if the image has a defect
    # Keep track of if the image was clipped during rotation
    # if the image was clipped it cannot be used
    clipped = False

    # Keep looking for a defect until we find one or give up
    # This is so we can get an even dataset even if defects do not appear
    # as 50 percent of the input image
    # Also we cant use a clipped image so if its clipped we keep looking
    while ((defect_type == no_defect_index or defect_type == -1) and tries < num_tries) or clipped:
        img = og_img.copy() # Copy the original image so we can modify our copy
        defect = og_defect.copy() # Copy the defect map so we can modify our copy

        # Inital trainsforms for both image and defect map
        # Convert them to the right format and grayscale
        img = TF.to_pil_image(img)
        img = TF.to_grayscale(img)
        defect = TF.to_pil_image(defect)

        # Set up other random rotation and apply to apply to both
        angle = random.randint(-180, 180)
        img = TF.rotate(img, angle)
        defect = TF.rotate(defect, angle)  # Cant use fill argument for TF.rotate()? pytorch too old?

        # Set up random croping and apply to both
        h = img_size[0]
        w = img_size[1]
        max_width, max_height = img.size
        top = random.randint(2*h, max_height - 2*h)
        left = random.randint(2*w, max_width - 2*w)
        img = TF.crop(img, top, left, h, w)
        defect = TF.crop(defect, top, left, h, w)

        # Apply a random (50% chance) horizontal flip to both
        if random.random() > 0.5:
            img = TF.hflip(img)
            defect = TF.hflip(defect)
        # Random vertical flipping (50% chance) and flip both
        if random.random() > 0.5:
            img = TF.vflip(img)
            defect = TF.vflip(defect)

        # If the input image contains a pure black it is (slmost certainly)
        # from being cliped during rotation, which is bad
        if img.getextrema()[0] == 0:
            clipped = True
        else:
            clipped = False


        # See what type of defect we have
        for j, c in enumerate(defect_colors):
            # See how many pixels we match all 3 channels to the test color
            # For now just match at least 10
            if torch.sum(torch.sum(TF.to_tensor(defect).permute(2,1,0) == torch.tensor(c), 2) == 4) > 9:
                defect_type = j
                break
        # Make sure we found a defect type
        if defect_type == -1 and not clipped:
            raise Exception("Error: Could not determine type for image", img)
        
        # Convert the images to tensors (different format)
        img = TF.to_tensor(img)
        defect = TF.to_tensor(defect)

        tries = tries + 1 # Increment number of tries

    # Once we exit the while loop we are ready to save our image into the correct folder
    # SAVE AS PNG! jpg has losses and it looks bad when it is loaded again later
    torchvision.utils.save_image(img, out_dir + os.sep + defect_names[defect_type] + os.sep + prefix + str(i) + ".png")
    num_defects[defect_type] = num_defects[defect_type] + 1

    # Print our progress
    print(str(round(i/num_imgs*100, 2)) + "%", end='\r')

# Print our final percentage of how many images had defects and how many didnt
# This way we can see if we our close to a 50/50 split that we want
for k, t in enumerate(defect_names):
    print(f"Generated {num_defects[k]} images with {t} defect ({100*num_defects[k]/num_imgs}%)")
