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

# How many images to generate and what size should they be
num_imgs = 3000
img_size = [150, 150]

# Path pointing to the folder that contains the large image and the defect map of the image
# The defect map should be pure black where the defects are and should be named IMG_NAME_defects.png
path = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/part/"
img = "part_train" # Without extension

# Where to save the output images
out_dir = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/part/sorted_training"

# Load the original image and the defect map
og_img = io.imread(path + os.path.sep + img + ".png")
og_defect = io.imread(path + os.path.sep + img + "_defects.png")

# Make directories to save the output images
os.makedirs(out_dir + "/defect/", exist_ok=True)
os.makedirs(out_dir + "/no_defect/", exist_ok=True)

num_defects = 0 # Running total of how many images we created had defects

# Loop to make all the images
for i in range(num_imgs): # I keeps track of image number so we can save them with unique names
    # num_ries is how many times to loop through, looking for a defect, before we
    # give up and just save a no-defect image. This should change relative to
    # the density of defects to get roughly a 50/50 split of defect and no-defect
    num_tries = 2
    # tries is the running counter of how many times we tried
    tries = 0

    defective = False # Keep track of if the image has a defect
    # Keep track of if the image was clipped during rotation
    # if the image was clipped it cannot be used
    clipped = False

    # Keep looking for a defect until we find one or give up
    # This is so we can get an even dataset even if defects do not appear
    # as 50 percent of the input image
    # Also we cant use a clipped image so if its clipped we keep looking
    while (not defective and tries < num_tries) or clipped:
        img = og_img.copy() # Copy the original image so we can modify our copy
        defect = og_defect.copy() # Copy the defect map so we can modify our copy

        # Inital trainsforms for both image and defect map
        # Convert them to the right format and grayscale
        img = TF.to_pil_image(img)
        img = TF.to_grayscale(img)
        defect = TF.to_pil_image(defect)
        defect = TF.to_grayscale(defect)


        # Set up other random rotation and apply to apply to both
        angle = random.randint(-180, 180)
        img = TF.rotate(img, angle)
        defect = TF.rotate(defect, angle) # Cant use fill argument for TF.rotate()? pytorch too old?

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


        # See if we have any defect in the defect map
        min_pix = defect.getextrema()[0] # Get the darkest pixel
        if(min_pix == 0): # If we have defects the dark pixel will be pure black
            defective = True
        else:
            defective = False

        # If the input image contains a pure black it is (slmost certainly)
        # from being cliped during rotation, which is bad
        if img.getextrema()[0] == 0:
            clipped = True
        else:
            clipped = False

        # Convert the images to tensors (different format)
        img = TF.to_tensor(img)
        defect = TF.to_tensor(defect)

        tries = tries + 1 # Increment number of tries

    # Once we exit the while loop we are ready to save our image into the correct folder
    # SAVE AS PNG! jpg has losses and it looks bad when it is loaded again later
    if defective: # If we have defects
        # Save the transformed image in the defects folder
        torchvision.utils.save_image(img, out_dir + os.sep + "defect" + os.sep + str(i) + ".png")
        num_defects = num_defects + 1
    else: # if we dont
        # Save the transformed image in the no_defects folder
        torchvision.utils.save_image(img, out_dir + os.sep + "no_defect" + os.sep + str(i) + ".png")

    # Print our progress
    print(str(round(i/num_imgs*100,2)) + "%", end='\r')

# Print our final percentage of how many images had defects and how many didnt
# This way we can see if we our close to a 50/50 split that we want
print("Genearated " + str(num_imgs) + " images, " + str(num_defects) + " (" + str(num_defects/num_imgs*100) + "%) had defects.")
