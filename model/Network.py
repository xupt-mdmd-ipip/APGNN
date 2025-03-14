import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

import imp
import scipy.io as io
import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
# from model import ViT


# ----------------------------------------------------------------------------------------------------
# model部分
def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer


def conv3x3x3_ft(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        FeatureWiseTransformation2d_fw(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer


class residual_block(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel, out_channel)
        self.conv2 = conv3x3x3(out_channel, out_channel)
        self.conv3 = conv3x3x3(out_channel, out_channel)

    def forward(self, x):  # (1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True)  # (1,8,100,9,9)  (1,16,25,5,5)
        x2 = F.relu(self.conv2(x1), inplace=True)  # (1,8,100,9,9) (1,16,25,5,5)
        x3 = self.conv3(x2)  # (1,8,100,9,9) (1,16,25,5,5)

        out = F.relu(x1 + x3, inplace=True)  # (1,8,100,9,9)  (1,16,25,5,5)
        return out


# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------

# --- feature-wise transformation layer ---这好像没有被使用到
def softplus(x):
    return torch.nn.functional.softplus(x, beta=100)


class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
    feature_augment = True

    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(FeatureWiseTransformation2d_fw, self).__init__(num_features, momentum=momentum,
                                                             track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        if self.feature_augment:  # initialize {gamma, beta} with {0.3, 0.5}
            self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1, 1) * 0.3)
            self.beta = torch.nn.Parameter(torch.ones(1, num_features, 1, 1, 1) * 0.5)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)

        # apply feature-wise transformation
        if self.feature_augment and self.training:
            gamma = (1 + torch.randn(1, self.num_features, 1, 1, 1, dtype=self.gamma.dtype,
                                     device=self.gamma.device) * softplus(self.gamma)).expand_as(out)
            beta = (torch.randn(1, self.num_features, 1, 1, 1, dtype=self.beta.dtype,
                                device=self.beta.device) * softplus(self.beta)).expand_as(out)
            out = gamma * out + beta
        return out


# ----------------------------------------------------------------------------------------------------


# 这是要添加的transformer的部分
# ----------------------------------------------------------------------------------------------------

def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        """
        nn.Sequential() 可以允许将整个容器视为单个模块（即相当于把多个模块封装成一个模块），
        forward()方法接收输入之后，nn.Sequential()按照内部模块的顺序自动依次计算并输出结果。
        这就意味着我们可以利用nn.Sequential() 自定义自己的网络层。
        """

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        # chunk(chunk数，维度），在此处的意思是将这个tensor-1维度切分为3块
        # 得到一个具有3个Tensor的元组

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # split into multi head attentions
        # （1）map函数是python中的一个内置函数，做映射。将函数映射作用于序列上
        # （2）map()函数返回的是一个新的迭代器对象，不会改变原有对象

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x


# ----------------------------------------------------------------------------------------------------


# 这是原本SSFTT中的网络
# ----------------------------------------------------------------------------------------------------

NUM_CLASS = 16


class SSFTTnet(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASS, num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8,
                 dropout=0.1, emb_dropout=0.1):
        super(SSFTTnet, self).__init__()
        self.L = num_tokens
        self.cT = dim

        # The number of 3D kernels is 8
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        # The number of 2D kernels is 64
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8 * 28, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, mask=None):
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        # print("3d",x.shape)
        x = self.conv2d_features(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        # print("2d",x.shape)

        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)  # main game
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)

        return x


# if __name__ == '__main__':
#     model = SSFTTnet()
#     model.eval()
#     print(model)
#     input = torch.randn(64, 1, 30, 13, 13)
#     y = model(input)
#     print(y.size())

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# 这是特征提取部分的核心网络函数
class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2, patch_size, emb_size):
        super(D_Res_3d_CNN, self).__init__()
        self.in_channel = in_channel
        self.emb_size = emb_size
        self.patch_size = patch_size
        self.block1 = residual_block(in_channel, out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4, 2, 2), padding=(0, 1, 1), stride=(4, 2, 2))
        self.block2 = residual_block(out_channel1, out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4, 2, 2), stride=(4, 2, 2), padding=(2, 1, 1))
        self.conv = nn.Conv3d(in_channels=out_channel2, out_channels=32, kernel_size=3, bias=False)

        self.layer_second = nn.Sequential(nn.Linear(in_features=self._get_layer_size()[0],
                                                    out_features=self.emb_size,
                                                    bias=True),
                                          nn.BatchNorm1d(self.emb_size))

        self.layer_last = nn.Sequential(nn.Linear(in_features=self._get_layer_size()[1],
                                                  out_features=self.emb_size,
                                                  bias=True),
                                        nn.BatchNorm1d(self.emb_size))

        self.pos_embedding = nn.Parameter(torch.empty(1, (4 + 1), 64))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 64))
        self.dropout = nn.Dropout(0.1)

        self.transformer = Transformer(64, 1, 8, 8, 0.1)



    def _get_layer_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, 100,
                             self.patch_size, self.patch_size))
            x = self.block1(x)
            x = self.maxpool1(x)
            x = self.block2(x)
            x = self.maxpool2(x)
            _, t, c, w, h = x.size()
            s1 = t * c * w * h
            x = self.conv(x)
            x = x.view(x.shape[0], -1)
            s2 = x.size()[1]
        return s1, s2

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.maxpool1(x)
        x = self.block2(x)
        x = self.maxpool2(x)
        inter = x
        inter = inter.view(inter.shape[0], -1)
        inter = self.layer_second(inter)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.layer_last(x)

        # x += self.pos_embedding
        # x = self.dropout(x)
        # x = self.transformer(x, None)  # main game
        out = []
        out.append(inter)
        out.append(x)
        # out = ViT.VisionTransformer()


        return out

    """
    xxxxxxxxxxxxxxxxxxx1 torch.Size([16, 100, 9, 9])
    xxxxxxxxxxxxxxxxxxx2 torch.Size([16, 1, 100, 9, 9])
    xxxxxxxxxxxxxxxxxxx3 torch.Size([16, 8, 100, 9, 9])
    xxxxxxxxxxxxxxxxxxx4 torch.Size([16, 8, 25, 5, 5])
    xxxxxxxxxxxxxxxxxxx5 torch.Size([16, 16, 25, 5, 5])
    xxxxxxxxxxxxxxxxxxx6 torch.Size([16, 16, 7, 3, 3])
    iiiiiiiiiiiiiiiiiii1 torch.Size([16, 1008])
    iiiiiiiiiiiiiiiiiii2 torch.Size([16, 64])
    xxxxxxxxxxxxxxxxxxx7 torch.Size([16, 32, 5, 1, 1])
    xxxxxxxxxxxxxxxxxxx8 torch.Size([16, 160])
    xxxxxxxxxxxxxxxxxxx9 torch.Size([16, 64])
    """
    # def forward(self, x):
    #     print('xxxxxxxxxxxxxxxxxxx1', x.shape)
    #     x = x.unsqueeze(1)
    #     print('xxxxxxxxxxxxxxxxxxx2', x.shape)
    #     x = self.block1(x)
    #     print('xxxxxxxxxxxxxxxxxxx3', x.shape)
    #     x = self.maxpool1(x)
    #     print('xxxxxxxxxxxxxxxxxxx4', x.shape)
    #     x = self.block2(x)
    #     print('xxxxxxxxxxxxxxxxxxx5', x.shape)
    #     x = self.maxpool2(x)
    #     print('xxxxxxxxxxxxxxxxxxx6', x.shape)
    #     inter = x
    #     inter = inter.view(inter.shape[0], -1)
    #     print('iiiiiiiiiiiiiiiiiii1', inter.shape)
    #     inter = self.layer_second(inter)
    #     print('iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii2', inter.shape)
    #     x = self.conv(x)
    #     print('xxxxxxxxxxxxxxxxxxx7', x.shape)
    #     x = x.view(x.shape[0], -1)
    #     print('xxxxxxxxxxxxxxxxxxx8', x.shape)
    #     x = self.layer_last(x)
    #     print('xxxxxxxxxxxxxxxxxxx9', x.shape)
    #     out = []
    #     # print('ooooooooooooooooooo1', out)
    #     out.append(inter)
    #     # print('ooooooooooooooooooo2', out)
    #     out.append(x)
    #     print('ooooooooooooooooooo3', out)
    #     return out


class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

    # def forward(self, x):
    #     print('xxxxxxxxxxxxxxxxxxx1', x.shape)
    #     x = self.preconv(x)
    #     print('xxxxxxxxxxxxxxxxxxx2', x.shape)
    #     x = self.preconv_bn(x)
    #     print('xxxxxxxxxxxxxxxxxxx3', x.shape)
    #     return x
    """
    ssssssssssssssssssssss torch.Size([16, 128, 9, 9])
    xxxxxxxxxxxxxxxxxxx1 torch.Size([16, 128, 9, 9])
    xxxxxxxxxxxxxxxxxxx2 torch.Size([16, 100, 9, 9])
    xxxxxxxxxxxxxxxxxxx3 torch.Size([16, 100, 9, 9])
    ssssssssssssssssssssss torch.Size([16, 100, 9, 9])
    
    ssssssssssssssssssssss torch.Size([304, 128, 9, 9])
    xxxxxxxxxxxxxxxxxxx1 torch.Size([304, 128, 9, 9])
    xxxxxxxxxxxxxxxxxxx2 torch.Size([304, 100, 9, 9])
    xxxxxxxxxxxxxxxxxxx3 torch.Size([304, 100, 9, 9])
    ssssssssssssssssssssss torch.Size([304, 100, 9, 9])
    源于中有128个波段，映射到100维度
    304 = 19 * 16
    19是源于的类别
    
    tttttttttttttttttttttt torch.Size([16, 200, 9, 9])
    xxxxxxxxxxxxxxxxxxx1 torch.Size([16, 200, 9, 9])
    xxxxxxxxxxxxxxxxxxx2 torch.Size([16, 100, 9, 9])
    xxxxxxxxxxxxxxxxxxx3 torch.Size([16, 100, 9, 9])
    tttttttttttttttttttttt torch.Size([16, 100, 9, 9])
    
    tttttttttttttttttttttt torch.Size([304, 200, 9, 9])
    xxxxxxxxxxxxxxxxxxx1 torch.Size([304, 200, 9, 9])
    xxxxxxxxxxxxxxxxxxx2 torch.Size([304, 100, 9, 9])
    xxxxxxxxxxxxxxxxxxx3 torch.Size([304, 100, 9, 9])
    tttttttttttttttttttttt torch.Size([304, 100, 9, 9])
    目标于中有200个维度，映射到100维度
    
"""


# class Network(nn.Module):
#     def __init__(self, patch_size, emb_size):
#         super(Network, self).__init__()
#         self.feature_encoder = D_Res_3d_CNN(1, 8, 16, patch_size, emb_size)
#         self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
#         self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)
#
#     def forward(self, x, domain='source'):
#         if domain == 'target':
#             x = self.target_mapping(x)  # (45, 100,9,9)
#         elif domain == 'source':
#             x = self.source_mapping(x)  # (45, 100,9,9)
#         feature = self.feature_encoder(x)  # (45, 64)
#         return feature

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    model = D_Res_3d_CNN(1, 8, 16, 9, 64)
    model.eval()
    print(model)
    # input = torch.randn(8, 16, 9, 64)
    # TypeError: __init__() missing 5 required positional arguments:
    # 'in_channel', 'out_channel1', 'out_channel2', 'patch_size', and 'emb_size'
    y = model(input)
    print(y.size())
