import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import time
import datetime
import sys
import torch
from torch.autograd import Variable
import torch
import numpy as np

from torchvision.utils import save_image,make_grid


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



#### CycleGAN used Last 50 Images to stabilize the trainning
class ReplayBuffer:
    '''
    CycleGAN uses Last 50 Images as buffer to stabilize the trainning process
    ReplayBuffer is a list that holds the last 50 generated images in a list

    '''

    def __init__(self, max_size=50):
        assert max_size > 0, "Warning: Trying to create an Empty buffer"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:  # Randomly replace/overwrite existing values
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        # Pad --> Conv2d (Same) --> IntanceNorm2d --> Relu --> Pad --> Conv2d (Same)  --> IntanceNorm2d
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)  # Adds the shortcut/bypass connection


# _____________________________________________________#
# ______________________Generator______________________#
# _____________________________________________________#

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks=9):
        '''
        input_shape : Tensor in (C,H,W) Format
        num_residual_blocks : Number of Residual blocks
        '''
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)  # Build Sequential model by list unpacking

    def forward(self, x):
        return self.model(x)


# _____________________________________________________#
# __________________Discriminator______________________#
# _____________________________________________________#

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        self.model = nn.Sequential(
            *self.discriminator_block(channels, 64, normalize=False),
            *self.discriminator_block(64, 128),
            *self.discriminator_block(128, 256),
            *self.discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),  #
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def discriminator_block(self, in_filters, out_filters, normalize=True):
        """Returns downsampling layers of each discriminator block"""
        layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, img):
        return self.model(img)