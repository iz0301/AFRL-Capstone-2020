# Script to sample and save images
import custom_nn as c
import torchvision

data_dir = "/home/isaac/Python/pytorch/AFRL-Capstone-2020/surface-defects/Defects"
IMSZ = 320

dataset = c.ImageDataset(data_dir, [IMSZ,IMSZ])

for i in range(len(dataset)):
    for j in range(1000):
        img, defect = dataset[i]
        torchvision.utils.save_image(img, data_dir + "/modified/" + str(i) + "_" + str(j) + ".jpg")
        torchvision.utils.save_image(defect, data_dir + "/modified/" + str(i) + "_" + str(j) + "_defects.jpg")
