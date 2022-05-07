from __future__ import print_function
import argparse
import os
import random
from typing import Dict
import imutils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as datas
import torchvision.transforms as tf
import torchvision.utils as tutils
from torch import Tensor
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torchvision.models as models


#### THE NETWORK ####

# Writing the VGG network
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        ##VGG layers

        # we chose average pooling since max pooling would only take the brightest values
        #   and min pooling would only take the darkest values.
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.p1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=False)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.p2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=False)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.p3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=False)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.p4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=False)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.p5 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=False)

    def forward(self, x, out_params=None):
        out: dict[str, Tensor] = {}

        out['re11'] = F.leaky_relu(self.conv1_1(x), 0.1, inplace=False)
        out['re12'] = F.leaky_relu(self.conv1_2(out['re11']), 0.1, inplace=False)
        out['p1'] = self.p1(out['re12'])
        # h_relu1_2 = out['re12']
        out['re21'] = F.leaky_relu(self.conv2_1(out['p1']), 0.1, inplace=False)
        out['re22'] = F.leaky_relu(self.conv2_2(out['re21']), 0.1, inplace=False)
        out['p2'] = self.p2(out['re22'])
        # h_relu2_2 = out['re22']
        out['re31'] = F.leaky_relu(self.conv3_1(out['p2']), 0.1, inplace=False)
        out['re32'] = F.leaky_relu(self.conv3_2(out['re31']), 0.1, inplace=False)
        out['re33'] = F.leaky_relu(self.conv3_3(out['re32']), 0.1, inplace=False)
        out['re34'] = F.leaky_relu(self.conv3_4(out['re33']), 0.1, inplace=False)
        out['p3'] = self.p3(out['re34'])
        # h_relu3_3 = out['re33']
        out['re41'] = F.leaky_relu(self.conv4_1(out['p3']), 0.1, inplace=False)
        out['re42'] = F.leaky_relu(self.conv4_2(out['re41']), 0.1, inplace=False)
        out['re43'] = F.leaky_relu(self.conv4_3(out['re42']), 0.1, inplace=False)
        out['re44'] = F.leaky_relu(self.conv4_4(out['re43']), 0.1, inplace=False)
        # h_relu4_3 = out['re43']
        out['p4'] = self.p4(out['re44'])
        out['re51'] = F.leaky_relu(self.conv5_1(out['p4']), 0.1, inplace=False)
        out['re52'] = F.leaky_relu(self.conv5_2(out['re51']), 0.1, inplace=False)
        out['re53'] = F.leaky_relu(self.conv5_3(out['re52']), 0.1, inplace=False)
        out['re54'] = F.leaky_relu(self.conv5_4(out['re53']), 0.1, inplace=False)
        out['p5'] = self.p5(out['re54'])

        h_relu1_2 = out['re12']
        h_relu2_2 = out['re22']
        h_relu3_3 = out['re33']
        h_relu4_3 = out['re43']

        if out_params is not None:
            return [out[param] for param in out_params]
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
