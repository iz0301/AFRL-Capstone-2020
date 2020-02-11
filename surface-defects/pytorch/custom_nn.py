import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import math
from skimage import io
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import time

# Class that I made to make creating convolutional nural network easier if you want i can add comments here too, just ask
class ConvolutionalNN(nn.Module):
    """
    input_size = [H,W,Nchannel] input image size
    conv_layers = n by 7 where:
        0 - n channels
        1 - kernel size
        2 - stride
        3 - padding
        4 - max pool kernel
        5 - max pool stride
        6 - max pool padding
    flat_layers = n by 1 where n is nubmer layers and value is nodes per layer
    n_output = number of output neurons
    activationFunc = activation function to use
    """
    def __init__(self, input_size, conv_layers, flat_layers, n_output, activationFunc):
        super(ConvolutionalNN,self).__init__()
        in_channels = int(input_size[2])
        d_size = [0,0]
        d_size[0] = input_size[0]
        d_size[1] = input_size[1]
        layers = nn.ModuleList()
        for c in range(conv_layers.shape[0]):
            out_channels = int(conv_layers[c,0])
            kernel = int(conv_layers[c,1])
            stride = int(conv_layers[c,2])
            padding = int(conv_layers[c,3])
            layers.append(nn.Conv2d(in_channels, out_channels, kernel, stride, padding))
            layers.append(activationFunc())

            # (W-F+2P)/S+1
            d_size[0] = math.floor((d_size[0] - kernel + 2*padding)/stride + 1)
            d_size[1] = math.floor((d_size[1] - kernel + 2*padding)/stride + 1)

            mk = int(conv_layers[c,4]) # max pool kernel
            ms = int(conv_layers[c,5]) # max pool stride
            mp = int(conv_layers[c,6]) # max pool padding
            layers.append(nn.MaxPool2d(mk,ms,mp))

            d_size[0] = math.floor(((d_size[0] - mk + 2*mp)/ms) + 1)
            d_size[1] = math.floor(((d_size[1] - mk + 2*mp)/ms) + 1)
            
            in_channels = out_channels # Input is last output

        layers.append(nn.Flatten())
        n_in = d_size[0] * d_size[1] * in_channels
        #print(d_size)
        #print(in_channels)
        for f in range(flat_layers.shape[0]):
            layers.append(nn.Linear(int(round(n_in)), flat_layers[f]))
            layers.append(activationFunc())
            n_in = flat_layers[f]

        layers.append(nn.Linear(n_in, n_output))
        layers.append(activationFunc())

        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def init_weights(self, mode):
        for i in range(len(self.layers)):
            if hasattr(self.layers[i],'weight'):
                for w in range(len(self.layers[i].weight)):
                    try:
                        mode(self.layers[i].weight[w])
                    except:
                        pass


    def get_features(self, x, num_layer):
        for i in range(num_layer):
            x = self.layers[i](x)
        return x
