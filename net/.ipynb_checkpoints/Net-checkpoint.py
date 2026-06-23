import numpy as np
import torch.fft
from thop import profile, clever_format
from torch import nn
import torch.nn.functional as F

from SHViT import DGWA, Conv2d_BN
from UNet import DoubleConvBlock
from attention import SEBlock, SCFE
import time
from thop import profile
import torch
import time

class cross_skips(nn.Module):
    def __init__(self, numbers=16, layers=4):
        super(cross_skips, self).__init__()
        self.layers = layers
        self.numbers = numbers
        self.skips = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        for i in range(layers):
            in_channels = numbers * (2 ** i)
            self.skips.append(SCFE(inc=in_channels))
            if i < layers - 1:
                self.conv_layers.append(
                    nn.Conv2d(numbers * (2 ** (i + 1)),  
                              numbers * (2 ** i), 
                              kernel_size=1
                              ))
    def forward(self, inputs):
        layer_outputs = [None] * self.layers

        for i in reversed(range(self.layers)):
            current_feature = inputs[i]
            current_out = self.skips[i](current_feature) + current_feature

            if i < self.layers - 1:
                higher_feature = layer_outputs[i + 1]
                higher_up = F.interpolate(higher_feature,size=current_out.shape[2:],mode='bilinear',align_corners=False)
                channel_adjust = self.conv_layers[i]
                higher_up = channel_adjust(higher_up)
                current_out = current_out + higher_up

            layer_outputs[i] = current_out
        return layer_outputs


class TCDGNet(nn.Module):
    def __init__(self,numbers=16):
        super(TCDGNet,self).__init__()
        self.numbers=numbers
        self.reduce_dim = nn.Conv2d(512, 384, kernel_size=1)
        self.en_layer0=nn.Sequential(
            DoubleConvBlock(3,self.numbers),
            DoubleConvBlock(self.numbers,self.numbers),
        )
        self.en_layer1 = nn.Sequential(
            DoubleConvBlock(self.numbers, 2*self.numbers),
            DoubleConvBlock(2*self.numbers, 2*self.numbers),
        )

        self.en_layer2 = nn.Sequential(
            DoubleConvBlock(2*self.numbers, 4 * self.numbers),
            DGWA(4 * self.numbers, 4 * self.numbers,r=0.125),
        )

        self.en_layer3 = nn.Sequential(
            DoubleConvBlock(4 * self.numbers, 8 * self.numbers),
            DGWA(8 * self.numbers, 8 * self.numbers,r=0.25),
        )

        self.en_layer4 = nn.Sequential(
            DoubleConvBlock(8 * self.numbers, 16 * self.numbers),
            DGWA(16 * self.numbers, 16 * self.numbers,r=0.5),
        )

        self.de_layer0 = nn.Sequential(
            DoubleConvBlock(3*self.numbers, self.numbers),
            DoubleConvBlock(self.numbers, self.numbers),
        )
        self.de_layer1 = nn.Sequential(
            DoubleConvBlock(6*self.numbers, 2 * self.numbers),
            DoubleConvBlock(2 * self.numbers, 2 * self.numbers),
        )
        self.de_layer2 = nn.Sequential(
            nn.Conv2d(12 * self.numbers, 4 * self.numbers, 3, 1, 1),
            nn.BatchNorm2d(4 * self.numbers),
            DGWA(4 * self.numbers, 4 * self.numbers,r=0.25),
        )
        self.de_layer3 = nn.Sequential(
            nn.Conv2d(24 * self.numbers, 8 * self.numbers,3,1,1),
            nn.BatchNorm2d(8 * self.numbers),
            DGWA(8 * self.numbers, 8 * self.numbers,r=0.5),
        )
        self.pool=nn.MaxPool2d(2,2)
        self.finalconv=nn.Conv2d(self.numbers,1,1,1,0)

        self.skips=cross_skips(numbers=self.numbers)
    def forward(self,x):
        en0=self.en_layer0(x)

        en1=self.pool(en0)
        en1=self.en_layer1(en1)

        en2 = self.pool(en1)
        en2 = self.en_layer2(en2)

        en3 = self.pool(en2)
        en3 = self.en_layer3(en3)
        
        en4 = self.pool(en3)
        en4 = self.en_layer4(en4)

        en0,en1,en2,en3=self.skips([en0,en1,en2,en3])

        de3 = F.interpolate(en4, scale_factor=2, mode='bilinear',align_corners=False)
        de3 = torch.cat([de3,en3],dim=1)
        de3 = self.de_layer3(de3)

        de2 = F.interpolate(de3, scale_factor=2, mode='bilinear', align_corners=False)
        de2 = torch.cat([de2, en2], dim=1)
        de2 = self.de_layer2(de2)

        de1 = F.interpolate(de2, scale_factor=2, mode='bilinear', align_corners=False)
        de1 = torch.cat([de1, en1], dim=1)
        de1 = self.de_layer1(de1)

        de0 = F.interpolate(de1, scale_factor=2, mode='bilinear', align_corners=False)
        de0 = torch.cat([de0, en0], dim=1)
        de0 = self.de_layer0(de0)

        out = self.finalconv(de0)
        return out