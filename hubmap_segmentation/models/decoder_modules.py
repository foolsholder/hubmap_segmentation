import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
import os
from typing import Dict


def conv3x3(in_channel, out_channel): #not change resolusion
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=3,stride=1,padding=1,dilation=1,bias=False)


def conv1x1(in_channel, out_channel): #not change resolution
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=1,stride=1,padding=0,dilation=1,bias=False)


def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform_(m.weight, gain=1)
        #nn.init.xavier_normal_(m.weight, gain=1)
        #nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1,0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            conv1x1(in_channel, in_channel//reduction).apply(init_weight),
            nn.ReLU(True),
            conv1x1(in_channel//reduction, in_channel).apply(init_weight)
        )

    def forward(self, inputs):
        x1 = self.global_maxpool(inputs)
        x2 = self.global_avgpool(inputs)
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x  = torch.sigmoid(x1 + x2)
        return x


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3 = conv3x3(2,1).apply(init_weight)

    def forward(self, inputs):
        x1,_ = torch.max(inputs, dim=1, keepdim=True)
        x2 = torch.mean(inputs, dim=1, keepdim=True)
        x  = torch.cat([x1,x2], dim=1)
        x  = self.conv3x3(x)
        x  = torch.sigmoid(x)
        return x


class CBAM(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channel, reduction)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, inputs):
        x = inputs * self.channel_attention(inputs)
        x = x * self.spatial_attention(x)
        return x


class CenterBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = conv3x3(in_channel, out_channel).apply(init_weight)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


class DecodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsample):
        super().__init__()
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample',nn.Upsample(scale_factor=2, mode='bilinear'))
        self.conv3x3_1 = conv3x3(in_channel, in_channel).apply(init_weight)
        self.bn1 = nn.BatchNorm2d(in_channel).apply(init_weight)

        self.conv3x3_2 = conv3x3(in_channel, out_channel).apply(init_weight)
        self.bn2 = nn.BatchNorm2d(out_channel).apply(init_weight)

        self.cbam = CBAM(out_channel, reduction=16)
        self.conv1x1   = conv1x1(in_channel, out_channel).apply(init_weight)
        self.bn_out = nn.BatchNorm2d(out_channel)

    def forward(self, inputs):
        # conv -> bn -> act
        skip_connection = self.upsample(inputs)

        #x  = F.relu(self.bn1(inputs))
        x  = skip_connection

        x  = F.relu(self.bn1(self.conv3x3_1(x)), inplace=True)

        x  = self.bn2(self.conv3x3_2(x))

        x  = self.cbam(x)

        x += self.conv1x1(skip_connection) #shortcut
        x = F.relu(self.bn_out(x), inplace=True)

        return x
