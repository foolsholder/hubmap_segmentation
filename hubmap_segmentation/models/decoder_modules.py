import torch
from torch import nn, optim
import math
import torch.nn.functional as F
import sys
import os
from typing import Dict, Optional


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


class DecodeBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            upsample,
            cls_emb_dim: int = 0
    ):
        super().__init__()
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample',nn.Upsample(scale_factor=2, mode='bilinear'))
        self.conv3x3_1 = conv3x3(in_channel, in_channel).apply(init_weight)
        self.bn1 = nn.BatchNorm2d(in_channel).apply(init_weight)

        self.conv3x3_2 = conv3x3(in_channel, out_channel).apply(init_weight)
        self.bn2 = nn.BatchNorm2d(out_channel).apply(init_weight)
        self.cbam = CBAM(out_channel, reduction=16)

        self.cls_emb_dim = cls_emb_dim
        if cls_emb_dim > 0:
            self.cls_emb_dense = nn.Sequential(
                nn.Linear(cls_emb_dim, out_channel),
                nn.BatchNorm1d(out_channel),
                nn.ReLU(inplace=True)
            )

    def forward(self, inputs, cls_emb: Optional[torch.Tensor] = None):
        # conv -> bn -> act
        skip_connection = self.upsample(inputs)

        #x  = F.relu(self.bn1(inputs))
        x  = skip_connection

        x  = F.relu(self.bn1(self.conv3x3_1(x)), inplace=True)

        x = self.bn2(self.conv3x3_2(x))

        if self.cls_emb_dim > 0:
            x += self.cls_emb_dense(cls_emb)[:, :, None, None]

        x  = self.cbam(x)
        x = F.relu(x)
        x += skip_connection #shortcut

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


from .ffc_modules import FFC_BN_ACT, FFCSE_block


class FFCDecodeBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            upsample,
            dilation=1,
            ratio_gin=0.5,
            ratio_gout=0.5,
            cls_emb_dim: int = 0,
            lfu=False,
            use_se=True
    ):
        super().__init__()
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample',nn.Upsample(scale_factor=2, mode='bilinear'))
        self.bn_out = nn.BatchNorm2d(out_channel)

        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv_bn_act1 = FFC_BN_ACT(in_channel, in_channel, kernel_size=3, padding=1,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout,
                                activation_layer=nn.ReLU, enable_lfu=lfu)
        self.conv_bn_act2 = FFC_BN_ACT(in_channel, out_channel, kernel_size=3,
                                ratio_gin=ratio_gout, ratio_gout=ratio_gout,
                                stride=1, padding=1, groups=1,
                                activation_layer=nn.ReLU, enable_lfu=lfu)
        self.conv_bn_act3 = FFC_BN_ACT(in_channel, out_channel, kernel_size=1,
                                ratio_gin=ratio_gout, ratio_gout=ratio_gout, enable_lfu=lfu)

        self.se_block = FFCSE_block(out_channel, ratio_gout) if use_se else nn.Identity()

    def forward(self, inputs, cls_emb: Optional[torch.Tensor] = None):
        skip_connection = self.upsample(inputs)
        x = skip_connection

        in_cg = self.conv_bn_act1.ffc.in_cg
        x_l, x_g = x[:, :-in_cg], x[:, -in_cg:]
        #print(x_l.shape, x_g.shape, flush=True)
        x = x_l, x_g
        skip_connection = x

        x = self.conv_bn_act1(x)
        x = self.conv_bn_act2(x)
        x = self.se_block(x)

        x_l, x_r = x
        sx_l, sx_r = self.conv_bn_act3(skip_connection)

        x = torch.cat([x_l, x_r], dim=1)
        skip = torch.cat([sx_l, sx_r], dim=1)

        x += skip
        return x


class FFCCenterBlock(FFCDecodeBlock):
    def __init__(self,
                 ratio_gin = 0.75,
                 ratio_gout = 0.75,
                 upsample=False,
                 **kwargs):
        super(FFCCenterBlock, self).__init__(
            ratio_gin=ratio_gin,
            ratio_gout=ratio_gout,
            upsample=upsample,
            **kwargs
        )


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):

    timesteps /= 6.

    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
