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

class ImageDataset(Dataset):
    # TODO:  Add transforms to modify the images for a bigger dataset
    # but do the actually coputation of transform when you get item
    # # TODO: Also load the corresponding map of the defects and
    # return a sort of a 2 part item in __getitem__
    def __init__(self, root_dir, img_size=[300,300]):
        self.root_dir = root_dir
        all_files = os.listdir(root_dir)
        self.img_files = []
        self.defect_maps = []
        for f in all_files:
            if not f.endswith('_defects.jpg'):
                self.img_files.append(f)
                self.defect_maps.append(f[0:-4] + '_defects.jpg')
        self.img_size = img_size
        self.color_jitter = transforms.ColorJitter()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):

        min_pix = 1
        lim = 0

        og_img = io.imread(self.root_dir + os.path.sep + self.img_files[idx])
        og_defect = io.imread(self.root_dir + os.path.sep + self.defect_maps[idx])
        while min_pix != 0 and lim < 15: # Loop until a defect is in sample
            #self.transform(io.imread(self.root_dir + os.path.sep + self.files[idx]))
#            st = time.time()
            img = og_img.copy()
            defect = og_defect.copy()

#            print("checkpoint 1 " + str(time.time() - st))
#            st = time.time()
            # Inital trainsforms for both image and defect map
            img = TF.to_pil_image(img)
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


            h = self.img_size[0]
            w = self.img_size[1]
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

            # Color jitter the test image
            img = self.color_jitter(img)

            # See if we have any defect in the defect map
            min_pix = defect.getextrema()[0]
            lim = lim + 1
            #print("End of while " + str(lim) + " minp = " + str(min_pix))

        img = TF.to_tensor(img)
        defect = TF.to_tensor(defect)
        return img, defect

class Autoencoder(nn.Module):
    """
    conv is a vector with number of feater maps per layer
    Output greyscale
    """
    def __init__(self, im_size, n_latent, conv, deconv):
        super(Autoencoder,self).__init__()

        conv_layers = np.empty([len(conv),7])
        deconv_layers = np.empty([len(deconv),5])
        for i in range(len(conv)):
            conv_layers[i] = [conv[i], 3, 1, 0, 2, 2, 0]
        for i in range(len(deconv)):
            deconv_layers[i] = [deconv[i], 3, 1, 0, 2]

        self.encoder = ConvolutionalNN(im_size, conv_layers, np.asarray([int(n_latent*1.0)]), n_latent, nn.Sigmoid)

        self.decoder = DeconvolutionalNN(n_latent, np.asarray([int(n_latent*2)]), deconv_layers, (im_size[0], im_size[1], 1), nn.Sigmoid)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    # Initalize wieghts from aonther Autoencoder network
    def initWeights(self, from_network):
        for i in range(len(self.encoder.layers)):
            if hasattr(self.encoder.layers[i],'weight'):
                for w in range(len(self.encoder.layers[i].weight)):
                    try:
                        self.encoder.layers[i].weight[w] = from_network.encoder.layers[i].weight[w].deepcopy()
                    except:
                        pass
                        #torch.nn.init.normal_(self.encoder.layers[i].weight[w])
        for i in range(len(self.decoder.convLayers)):
            if hasattr(self.decoder.convLayers[i],'weight'):
                for w in range(len(self.decoder.convLayers[i].weight)):
                    try:
                        self.decoder.convLayers[i].weight[w] = from_network.decoder.convLayers[i].weight[w].deepcopy()
                    except:
                        pass
                    #    torch.nn.init.normal_(self.decoder.convLayers[i].weight[w])
        for i in range(len(self.decoder.flatLayers)):
            if hasattr(self.decoder.flatLayers[i],'weight'):
                for w in range(len(self.decoder.flatLayers[i].weight)):
                    try:
                        self.decoder.flatLayers[i].weight[w] = from_network.decoder.flatLayers[i].weight[w].deepcopy()
                    except:
                        pass
                        #torch.nn.init.normal_(self.decoder.flatLayers[i].weight[w])

class DeconvolutionalNN(nn.Module):
    """
    n_in = number of input nodes
    flat_layers = n length where each value is # neurons
    conv_layers = n by 5 where:
        0 - channels
        1 - kernel
        2 - stride
        3 - padding
        4 - upscale factor
    out_size = desiered shape of output HxWxNchannels
    """
    def __init__(self, n_in, flat_layers, conv_layers, out_size, activationFunc):
        super(DeconvolutionalNN,self).__init__()
        nn_flat_layers = nn.ModuleList()

        for f in range(flat_layers.shape[0]):
            nn_flat_layers.append(nn.Linear(n_in, flat_layers[f]))
            nn_flat_layers.append(activationFunc())
            n_in = flat_layers[f]

        self.flatLayers = nn_flat_layers

        in_channels = int(round(n_in))

        nn_conv_layers = nn.ModuleList()


        in_channels = int(round(in_channels/(5*5)))
        self.n_ch = in_channels

        for c in range(conv_layers.shape[0]):
            out_channels = int(round(conv_layers[c,0]))
            kernel = int(round(conv_layers[c,1]))
            stride = int(round(conv_layers[c,2]))
            padding = int(round(conv_layers[c,3]))

            us = float(conv_layers[c,4]) # upscale factor
            nn_conv_layers.append(nn.Upsample(scale_factor=us))

            nn_conv_layers.append(nn.ConvTranspose2d(in_channels, out_channels,
                kernel, stride, padding))
            nn_conv_layers.append(activationFunc())

            in_channels = out_channels # Input is last output

        nn_conv_layers.append(nn.ConvTranspose2d(in_channels, out_size[2], kernel, stride, padding))
        nn_conv_layers.append(activationFunc())
        nn_conv_layers.append(nn.Upsample(size=[out_size[0], out_size[1]]))
        self.convLayers = nn_conv_layers
        self.upsampler = nn.Upsample(size=[10, 10])

    def forward(self, x):
        for flat in self.flatLayers:
            x = flat(x)
        x = x.view(-1, self.n_ch, 5, 5)
        #x = self.upsampler(x)
        for conv in self.convLayers:
            x = conv(x)
        return x


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

            # (𝑊−𝐹+2𝑃)/𝑆+1
            d_size[0] = math.floor((d_size[0] - kernel + 2*padding)/stride + 1)
            d_size[1] = math.floor((d_size[1] - kernel + 2*padding)/stride + 1)

            mk = int(conv_layers[c,4]) # max pool kernel
            ms = int(conv_layers[c,5]) # max pool stride
            mp = int(conv_layers[c,6]) # max pool padding
            layers.append(nn.MaxPool2d(mk,ms,mp))

            d_size[0] = math.floor((d_size[0] - mk + 2*mp)/ms + 1)
            d_size[1] = math.floor((d_size[1] - mk + 2*mp)/ms + 1)

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
