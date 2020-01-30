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
import matplotlib.pyplot as plt


def show_defects(img, defect_map, threshold=0.5):
    f = plt.figure(1)
    plt.imshow(img, cmap="gray")
    #plt.hold(True)
    map = plt.imshow(np.array(defect_map < threshold, dtype=float), cmap="Reds", alpha=0.3)

    map_on = True
    while plt.fignum_exists(1):
        plt.ginput()
        map_on = not map_on
        if map_on:
            map.set_data(np.array(defect_map < threshold, dtype=float))
        else:
            map.set_data(np.zeros_like(defect_map, dtype=float))

    plt.show()

"""
Test example:
img = np.array([[1,2,3,4,5,6,7,8,9,10],[2,3,4,5,6,7,8,9,10,11],[3,4,5,6,7,8,9,10,11,12],[4,5,6,7,8,9,10,11,12,13]])

dm = np.zeros_like(img)
dm[1:3,1:3] = 1

show_defects(img, dm)
"""
