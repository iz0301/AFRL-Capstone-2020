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


num_imgs = 1000
img_size = [150, 150]
path = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/"
img = "IMG_1278" # Without extension
out_dir = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects/sorted"

og_img = io.imread(path + os.path.sep + img + ".jpg")
og_defect = io.imread(path + os.path.sep + img + "_defects.jpg")

os.makedirs(out_dir + "/defect/", exist_ok=True)
os.makedirs(out_dir + "/no_defect/", exist_ok=True)
num_defects = 0

for i in range(num_imgs):
    # So randomly originally ~8% had defects so we will try to increase that by trying for defects many (10) times before giving up
    tries = 0
    defective = False
    clipped = False
    while (not defective and tries < 8) or clipped:
        img = og_img.copy()
        defect = og_defect.copy()

    #            print("checkpoint 1 " + str(time.time() - st))
    #            st = time.time()
        # Inital trainsforms for both image and defect map
        img = TF.to_pil_image(img)
        img = TF.to_grayscale(img)
        defect = TF.to_pil_image(defect)
        defect = TF.to_grayscale(defect)

    #            print("checkpoint 2 " + str(time.time() - st))
    #            st = time.time()

        # Set up other random transforms to apply to both
        angle = random.randint(-180, 180)
        img = TF.rotate(img, angle)
        defect = TF.rotate(defect, angle) # Cant use fill argument? pytorch too old?
    #            print("checkpoint 3 " + str(time.time() - st))
    #            st = time.time()

        h = img_size[0]
        w = img_size[1]
        max_width, max_height = img.size
        top = random.randint(2*h, max_height - 2*h)
        left = random.randint(2*w, max_width - 2*w)
        #crop(img, top, left, height, width)
        img = TF.crop(img, top, left, h, w)
        defect = TF.crop(defect, top, left, h, w)
    #            print("checkpoint 4 " + str(time.time() - st))
    #            st = time.time()

        if random.random() > 0.5:
            img = TF.hflip(img)
            defect = TF.hflip(defect)

        # Random vertical flipping
        if random.random() > 0.5:
            img = TF.vflip(img)
            defect = TF.vflip(defect)

    #            print("checkpoint 5 " + str(time.time() - st))
    #            st = time.time()

        # See if we have any defect in the defect map
        min_pix = defect.getextrema()[0]
        if(min_pix == 0): # If we have defects
            defective = True
        else:
            defective = False
        if img.getextrema()[0] == 0: # If the image contains a pure black it is from rotation which is bad
            clipped = True
        else:
            clipped = False

        img = TF.to_tensor(img)
        defect = TF.to_tensor(defect)
        tries = tries + 1
    if defective: # If we have defects
        torchvision.utils.save_image(img, out_dir + "/defect/" + str(i) + ".jpg")
        num_defects = num_defects + 1
    else: # if we dont
        torchvision.utils.save_image(img, out_dir + "/no_defect/" + str(i) + ".jpg")
    print(str(round(i/num_imgs*100,2)) + "%", end='\r')
print("Genearated " + str(num_imgs) + " images, " + str(num_defects) + " (" + str(num_defects/num_imgs*100) + "%) had defects.")
