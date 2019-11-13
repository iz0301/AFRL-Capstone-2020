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

def train(model: nn.Module, lossFunc, optimizer, num_epochs, dataloader: DataLoader, device: str):
    for epoch in range(num_epochs):
        #end_time = time.time()
        for batch_num, (img, defect) in enumerate(dataloader):

            #start_time = time.time()
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

            #print("Batch took " + str(time.time() - start_time))
            #end_time = time.time()
        #print("epoch [%d/%d], loss:{%.4f}" % (epoch+1, num_epochs, loss.data), flush=True)
    return loss.data
