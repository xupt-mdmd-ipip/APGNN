import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class attention1d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention1d, self).__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv1d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv1d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)


class Dynamic_conv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, K=4, temperature=34, init_weight=True):
        super(Dynamic_conv1d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention1d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // groups, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):  # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height = x.size()
        x = x.view(1, -1, height, )  # 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size * self.out_planes,
                                                                    self.in_planes // self.groups, self.kernel_size, )
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv1d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv1d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-1))
        return output


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, K=4, temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):  # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)  # 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size * self.out_planes,
                                                                    self.in_planes // self.groups, self.kernel_size,
                                                                    self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output


class attention3d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature):
        super(attention3d, self).__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv3d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv3d(hidden_planes, K, 1, bias=False)
        # self.fc1 = nn.Conv3d(1, 8, 1, bias=False)
        # self.fc2 = nn.Conv3d(8, K, 1, bias=False)
        self.temperature = temperature

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        print("1", x.shape)
        x = self.avgpool(x)
        print("1", x.shape)
        x = self.fc1(x)
        print("2", x.shape)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)


class Dynamic_conv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, K=4, temperature=34):
        super(Dynamic_conv3d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention3d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(
            torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None

        # TODO 初始化
        # nn.init.kaiming_uniform_(self.weight, )

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):  # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        print(x.shape)
        softmax_attention = self.attention(x)
        print(softmax_attention.shape)
        batch_size, in_planes, depth, height, width = x.size()
        x = x.view(1, -1, depth, height, width)  # 变化成一个维度进行组卷积
        print("yige", x.shape)
        weight = self.weight.view(self.K, -1)
        print(weight.shape)
        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size * self.out_planes,
                                                                    self.in_planes // self.groups,
                                                                    self.kernel_size, self.kernel_size,
                                                                    self.kernel_size)

        print("9999", aggregate_weight.shape)
        if self.bias is not None:
            print(x.shape)
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv3d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            print(output.shape)
        else:

            output = F.conv3d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-3), output.size(-2), output.size(-1))
        return output

#
# if __name__ == '__main__':
#     x = torch.randn(80, 30, 1, 15, 15)
#     model = Dynamic_conv3d(in_planes=30, out_planes=60, kernel_size=7, ratio=0.25, padding=3, )
#     # x = x.to('cuda:0')
#     # model.to('cuda')
#     # model.attention.cuda()
#     # nn.Conv3d()
#     print(model(x).shape)
#     # model.update_temperature()
#     # model.update_temperature()
#     # model.update_temperature()
#     # model.update_temperature()
#     # model.update_temperature()
#     # model.update_temperature()
#     # model.update_temperature()
#     # model.update_temperature()
#     # model.update_temperature()
#     # model.update_temperature()
#     # model.update_temperature()
#     # model.update_temperature()
#     # model.update_temperature()
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)
#     # print(model(x).shape)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)

        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Conv3d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm3d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv3d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv3d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv3d(attention_channel, kernel_size * kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv3d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):

        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):

        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size,
                                                    self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):

        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv3d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size, kernel_size),
            requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.

        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, depth, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, 1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size, self.kernel_size])
        output = F.conv3d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-3), output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)


def odconv3x3x3(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num)


def odconv5x5x5(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv3d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2,
                    reduction=reduction, kernel_num=kernel_num)


def odconv7x7x7(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv3d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3,
                    reduction=reduction, kernel_num=kernel_num)


# from odconv import ODConv2d
#
#
# def odconv3x3(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
#     return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0,
#                     reduction=reduction, kernel_num=kernel_num)
#
#
# def odconv5x5(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
#     return ODConv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=0,
#                     reduction=reduction, kernel_num=kernel_num)
#
#
# def odconv7x7(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
#     return ODConv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=0,
#                     reduction=reduction, kernel_num=kernel_num)
#
#
# class BasicBlock111(nn.Module):
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
#         super(BasicBlock111, self).__init__()
#
#         self.conv3D_3_1 = odconv3x3x3(inplanes, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn1 = nn.BatchNorm3d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_3_2 = odconv3x3x3(60, 120, reduction=reduction, kernel_num=kernel_num)
#         self.bn2 = nn.BatchNorm3d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_3_1 = odconv3x3(120, 120, reduction=reduction, kernel_num=kernel_num)
#         self.bn3 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_3_2 = odconv3x3(120, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn4 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_5_1 = odconv5x5x5(30, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn5 = nn.BatchNorm3d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_5_2 = odconv5x5x5(60, 120, reduction=reduction, kernel_num=kernel_num)
#         self.bn6 = nn.BatchNorm3d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_5_1 = odconv5x5(120, 120, reduction=reduction, kernel_num=kernel_num)
#         self.bn7 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_5_2 = odconv5x5(120, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn8 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_7_1 = odconv7x7x7(30, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn9 = nn.BatchNorm3d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_7_2 = odconv7x7x7(60, 120, reduction=reduction, kernel_num=kernel_num)
#         self.bn10 = nn.BatchNorm3d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_7_1 = odconv7x7(120, 120, reduction=reduction, kernel_num=kernel_num)
#         self.bn11 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_7_2 = odconv7x7(120, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn12 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.TripletAttention = TripletAttention()
#         self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc3 = nn.Linear(180, 16)
#
#     def forward(self, x):
#         out1 = self.conv3D_3_1(x)
#         out1 = self.bn1(out1)
#         out1 = self.relu(out1)
#
#         out11 = self.conv3D_3_2(out1)
#         out11 = self.bn2(out11)
#         out11 = self.relu(out11)
#
#         out11 = out11.reshape(out11.shape[0], -1, 19, 19)
#         out111 = self.conv2D_3_1(out11)
#         out111 = self.bn3(out111)
#         out111 = self.relu(out111)
#
#         out1111 = self.conv2D_3_2(out111)
#         out1111 = self.bn4(out1111)
#         out1111 = self.relu(out1111)
#         out1111 = self.TripletAttention(out1111)
#         # print("7999",out1111.shape)
#         ###第二层
#
#         out2 = self.conv3D_5_1(x)
#         out2 = self.bn5(out2)
#         out2 = self.relu(out2)
#         # out2 = torch.cat((out1, out2), dim=1)
#
#         out22 = self.conv3D_5_2(out2)
#         out22 = self.bn6(out22)
#         out22 = self.relu(out22)
#
#         out22 = out22.reshape(out22.shape[0], -1, 19, 19)
#
#         out222 = self.conv2D_5_1(out22)
#         out222 = self.bn7(out222)
#         out222 = self.relu(out222)
#         # out222 = torch.cat((out111, out222), dim=1)
#
#         out2222 = self.conv2D_5_2(out222)
#         out2222 = self.bn8(out2222)
#         out2222 = self.relu(out2222)
#         out2222 = self.TripletAttention(out2222)
#         # print("8999", out2222.shape)
#
#         ###第三层
#
#         out3 = self.conv3D_7_1(x)
#         out3 = self.bn9(out3)
#         out3 = self.relu(out3)
#         # out3 = torch.cat((out2, out3), dim=1)
#
#         out33 = self.conv3D_7_2(out3)
#         out33 = self.bn10(out33)
#         out33 = self.relu(out33)
#         out33 = out33.reshape(out33.shape[0], -1, 19, 19)
#
#         out333 = self.conv2D_7_1(out33)
#         out333 = self.bn11(out333)
#         out333 = self.relu(out333)
#         # out333 = torch.cat((out222, out333), dim=1)
#
#         out3333 = self.conv2D_7_2(out333)
#         out3333 = self.bn12(out3333)
#         out3333 = self.relu(out3333)
#         out3333 = self.TripletAttention(out3333)
#         # print("9999", out3333.shape)
#
#         out1 = self.avgpool3(out1111)
#         out4 = self.avgpool3(out2222)
#         out8 = self.avgpool3(out3333)
#
#         out1 = out1.reshape(out1.shape[0], -1)
#         out4 = out4.reshape(out4.shape[0], -1)
#         out8 = out8.reshape(out8.shape[0], -1)
#
#         out = torch.cat((out8, out4, out1), dim=1)
#         out = self.fc3(out)
#
#         return out
#
#
# class BasicBlock111_wuzhuyili(nn.Module):
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
#         super(BasicBlock111_wuzhuyili, self).__init__()
#
#         self.conv3D_3_1 = odconv3x3x3(inplanes, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn1 = nn.BatchNorm3d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_3_2 = odconv3x3x3(60, 120, reduction=reduction, kernel_num=kernel_num)
#         self.bn2 = nn.BatchNorm3d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_3_1 = odconv3x3(120, 180, reduction=reduction, kernel_num=kernel_num)
#         self.bn3 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_3_2 = odconv3x3(180, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn4 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_5_1 = odconv5x5x5(30, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn5 = nn.BatchNorm3d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_5_2 = odconv5x5x5(60, 120, reduction=reduction, kernel_num=kernel_num)
#         self.bn6 = nn.BatchNorm3d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_5_1 = odconv5x5(120, 180, reduction=reduction, kernel_num=kernel_num)
#         self.bn7 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_5_2 = odconv5x5(180, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn8 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_7_1 = odconv7x7x7(30, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn9 = nn.BatchNorm3d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_7_2 = odconv7x7x7(60, 120, reduction=reduction, kernel_num=kernel_num)
#         self.bn10 = nn.BatchNorm3d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_7_1 = odconv7x7(120, 180, reduction=reduction, kernel_num=kernel_num)
#         self.bn11 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_7_2 = odconv7x7(180, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn12 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.TripletAttention = TripletAttention()
#         self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc3 = nn.Linear(180, 16)
#
#     def forward(self, x):
#         out1 = self.conv3D_3_1(x)
#         out1 = self.bn1(out1)
#         out1 = self.relu(out1)
#
#         out11 = self.conv3D_3_2(out1)
#         out11 = self.bn2(out11)
#         out11 = self.relu(out11)
#
#         out11 = out11.reshape(out11.shape[0], -1, 15, 15)
#         out111 = self.conv2D_3_1(out11)
#         out111 = self.bn3(out111)
#         out111 = self.relu(out111)
#
#         out1111 = self.conv2D_3_2(out111)
#         out1111 = self.bn4(out1111)
#         out1111 = self.relu(out1111)
#         out1111 = self.TripletAttention(out1111)
#         # print("7999",out1111.shape)
#         ###第二层
#
#         out2 = self.conv3D_5_1(x)
#         out2 = self.bn5(out2)
#         out2 = self.relu(out2)
#         # out2 = torch.cat((out1, out2), dim=1)
#
#         out22 = self.conv3D_5_2(out2)
#         out22 = self.bn6(out22)
#         out22 = self.relu(out22)
#
#         out22 = out22.reshape(out22.shape[0], -1, 15, 15)
#
#         out222 = self.conv2D_5_1(out22)
#         out222 = self.bn7(out222)
#         out222 = self.relu(out222)
#         # out222 = torch.cat((out111, out222), dim=1)
#
#         out2222 = self.conv2D_5_2(out222)
#         out2222 = self.bn8(out2222)
#         out2222 = self.relu(out2222)
#         out2222 = self.TripletAttention(out2222)
#         # print("8999", out2222.shape)
#
#         ###第三层
#
#         out3 = self.conv3D_7_1(x)
#         out3 = self.bn9(out3)
#         out3 = self.relu(out3)
#         # out3 = torch.cat((out2, out3), dim=1)
#
#         out33 = self.conv3D_7_2(out3)
#         out33 = self.bn10(out33)
#         out33 = self.relu(out33)
#         out33 = out33.reshape(out33.shape[0], -1, 15, 15)
#
#         out333 = self.conv2D_7_1(out33)
#         out333 = self.bn11(out333)
#         out333 = self.relu(out333)
#         # out333 = torch.cat((out222, out333), dim=1)
#
#         out3333 = self.conv2D_7_2(out333)
#         out3333 = self.bn12(out3333)
#         out3333 = self.relu(out3333)
#         out3333 = self.TripletAttention(out3333)
#         # print("9999", out3333.shape)
#
#         out1 = self.avgpool3(out1111)
#         out4 = self.avgpool3(out2222)
#         out8 = self.avgpool3(out3333)
#
#         out1 = out1.reshape(out1.shape[0], -1)
#         out4 = out4.reshape(out4.shape[0], -1)
#         out8 = out8.reshape(out8.shape[0], -1)
#
#         out = torch.cat((out8, out4, out1), dim=1)
#         out = self.fc3(out)
#
#         return out
#
#
# class BasicBlock111_jingtai_douyou(nn.Module):
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
#         super(BasicBlock111_jingtai_douyou, self).__init__()
#
#         self.conv3D_3_1 = nn.Conv3d(inplanes, 60, kernel_size=(3, 3, 3), stride=1, padding=1)
#         self.bn1 = nn.BatchNorm3d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_3_2 = nn.Conv3d(60, 120, kernel_size=(3, 3, 3), stride=1, padding=1)
#         self.bn2 = nn.BatchNorm3d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_3_1 = nn.Conv2d(120, 180, kernel_size=(3, 3), stride=1, padding=0)
#         self.bn3 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_3_2 = nn.Conv2d(180, 60, kernel_size=(3, 3), stride=1, padding=0)
#         self.bn4 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_5_1 = nn.Conv3d(30, 60, kernel_size=(5, 5, 5), stride=1, padding=2)
#         self.bn5 = nn.BatchNorm3d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_5_2 = nn.Conv3d(60, 120, kernel_size=(5, 5, 5), stride=1, padding=2)
#         self.bn6 = nn.BatchNorm3d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_5_1 = nn.Conv2d(120, 180, kernel_size=5, stride=1, padding=0)
#         self.bn7 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_5_2 = nn.Conv2d(180, 60, kernel_size=5, stride=1, padding=0)
#         self.bn8 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_7_1 = nn.Conv3d(30, 60, kernel_size=(7, 7, 7), stride=1, padding=3)
#         self.bn9 = nn.BatchNorm3d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_7_2 = nn.Conv3d(60, 120, kernel_size=(7, 7, 7), stride=1, padding=3)
#         self.bn10 = nn.BatchNorm3d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_7_1 = nn.Conv2d(120, 180, kernel_size=(7, 7), stride=1, padding=0)
#         self.bn11 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_7_2 = nn.Conv2d(180, 60, kernel_size=(7, 7), stride=1, padding=0)
#         self.bn12 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.TripletAttention = TripletAttention()
#         self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc3 = nn.Linear(180, 16)
#
#     def forward(self, x):
#         out1 = self.conv3D_3_1(x)
#         out1 = self.bn1(out1)
#         out1 = self.relu(out1)
#
#         out11 = self.conv3D_3_2(out1)
#         out11 = self.bn2(out11)
#         out11 = self.relu(out11)
#
#         out11 = out11.reshape(out11.shape[0], -1, 15, 15)
#         out111 = self.conv2D_3_1(out11)
#         out111 = self.bn3(out111)
#         out111 = self.relu(out111)
#
#         out1111 = self.conv2D_3_2(out111)
#         out1111 = self.bn4(out1111)
#         out1111 = self.relu(out1111)
#         out1111 = self.TripletAttention(out1111)
#         # print("7999",out1111.shape)
#         ###第二层
#
#         out2 = self.conv3D_5_1(x)
#         out2 = self.bn5(out2)
#         out2 = self.relu(out2)
#         # out2 = torch.cat((out1, out2), dim=1)
#
#         out22 = self.conv3D_5_2(out2)
#         out22 = self.bn6(out22)
#         out22 = self.relu(out22)
#
#         out22 = out22.reshape(out22.shape[0], -1, 15, 15)
#
#         out222 = self.conv2D_5_1(out22)
#         out222 = self.bn7(out222)
#         out222 = self.relu(out222)
#         # out222 = torch.cat((out111, out222), dim=1)
#
#         out2222 = self.conv2D_5_2(out222)
#         out2222 = self.bn8(out2222)
#         out2222 = self.relu(out2222)
#         out2222 = self.TripletAttention(out2222)
#         # print("8999", out2222.shape)
#
#         ###第三层
#
#         out3 = self.conv3D_7_1(x)
#         out3 = self.bn9(out3)
#         out3 = self.relu(out3)
#         # out3 = torch.cat((out2, out3), dim=1)
#
#         out33 = self.conv3D_7_2(out3)
#         out33 = self.bn10(out33)
#         out33 = self.relu(out33)
#         out33 = out33.reshape(out33.shape[0], -1, 15, 15)
#
#         out333 = self.conv2D_7_1(out33)
#         out333 = self.bn11(out333)
#         out333 = self.relu(out333)
#         # out333 = torch.cat((out222, out333), dim=1)
#
#         out3333 = self.conv2D_7_2(out333)
#         out3333 = self.bn12(out3333)
#         out3333 = self.relu(out3333)
#         out3333 = self.TripletAttention(out3333)
#         # print("9999", out3333.shape)
#
#         out1 = self.avgpool3(out1111)
#         out4 = self.avgpool3(out2222)
#         out8 = self.avgpool3(out3333)
#
#         out1 = out1.reshape(out1.shape[0], -1)
#         out4 = out4.reshape(out4.shape[0], -1)
#         out8 = out8.reshape(out8.shape[0], -1)
#
#         out = torch.cat((out8, out4, out1), dim=1)
#         out = self.fc3(out)
#
#         return out
#
#
# from fcanet import MultiSpectralAttentionLayer
#
#
# class BasicBlock111_jingtai(nn.Module):
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
#         super(BasicBlock111_jingtai, self).__init__()
#
#         self.conv3D_3_1 = nn.Conv2d(inplanes, 60, kernel_size=(3, 3), stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_3_2 = nn.Conv2d(60, 120, kernel_size=(3, 3), stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_3_1 = nn.Conv2d(120, 180, kernel_size=(3, 3), stride=1, padding=0)
#         self.bn3 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_3_2 = nn.Conv2d(180, 60, kernel_size=(3, 3), stride=1, padding=0)
#         self.bn4 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_5_1 = nn.Conv2d(30, 60, kernel_size=(3, 3), stride=1, padding=1)
#         self.bn5 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_5_2 = nn.Conv2d(60, 120, kernel_size=(3, 3), stride=1, padding=1)
#         self.bn6 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_5_1 = nn.Conv2d(120, 180, kernel_size=3, stride=1, padding=0)
#         self.bn7 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_5_2 = nn.Conv2d(180, 60, kernel_size=3, stride=1, padding=0)
#         self.bn8 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_7_1 = nn.Conv2d(30, 60, kernel_size=(3, 3), stride=1, padding=1)
#         self.bn9 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_7_2 = nn.Conv2d(60, 120, kernel_size=(3, 3), stride=1, padding=1)
#         self.bn10 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_7_1 = nn.Conv2d(120, 180, kernel_size=(3, 3), stride=1, padding=0)
#         self.bn11 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_7_2 = nn.Conv2d(180, 60, kernel_size=(3, 3), stride=1, padding=0)
#         self.bn12 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         # c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
#         # self.avgpool3 = MultiSpectralAttentionLayer(channel=60, dct_h=c2wh[256], dct_w=c2wh[256], reduction=16,
#         #                                             freq_sel_method='top16')  # fca(x)
#         self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc3 = nn.Linear(180, 16)
#
#     def forward(self, x):
#         out1 = self.conv3D_3_1(x)
#         out1 = self.bn1(out1)
#         out1 = self.relu(out1)
#
#         # out11 = self.conv3D_3_2(out1)
#         # out11 = self.bn2(out11)
#         # out11 = self.relu(out11)
#         #
#         #
#         # out111 = self.conv2D_3_1(out11)
#         # out111 = self.bn3(out111)
#         # out111 = self.relu(out111)
#         #
#         # out1111 = self.conv2D_3_2(out111)
#         # out1111 = self.bn4(out1111)
#         # out1111 = self.relu(out1111)
#
#         ###第二层
#
#         out2 = self.conv3D_5_1(x)
#         out2 = self.bn5(out2)
#         out2 = self.relu(out2)
#         # out2 = torch.cat((out1, out2), dim=1)
#
#         # out22 = self.conv3D_5_2(out2)
#         # out22 = self.bn6(out22)
#         # out22 = self.relu(out22)
#         #
#         #
#         #
#         #
#         # out222 = self.conv2D_5_1(out22)
#         # out222 = self.bn7(out222)
#         # out222 = self.relu(out222)
#         # # out222 = torch.cat((out111, out222), dim=1)
#         #
#         # out2222 = self.conv2D_5_2(out222)
#         # out2222 = self.bn8(out2222)
#         # out2222 = self.relu(out2222)
#
#         ###第三层
#
#         out3 = self.conv3D_7_1(x)
#         out3 = self.bn9(out3)
#         out3 = self.relu(out3)
#         # out3 = torch.cat((out2, out3), dim=1)
#
#         # out33 = self.conv3D_7_2(out3)
#         # out33 = self.bn10(out33)
#         # out33 = self.relu(out33)
#         #
#         #
#         #
#         # out333 = self.conv2D_7_1(out33)
#         # out333 = self.bn11(out333)
#         # out333 = self.relu(out333)
#         # # out333 = torch.cat((out222, out333), dim=1)
#         #
#         # out3333 = self.conv2D_7_2(out333)
#         # out3333 = self.bn12(out3333)
#         # out3333 = self.relu(out3333)
#
#         out1 = self.avgpool3(out1)
#         out4 = self.avgpool3(out2)
#         out8 = self.avgpool3(out3)
#
#         out1 = out1.reshape(out1.shape[0], -1)
#         out4 = out4.reshape(out4.shape[0], -1)
#         out8 = out8.reshape(out8.shape[0], -1)
#
#         out = torch.cat((out8, out4, out1), dim=1)
#         out = self.fc3(out)
#
#         return out
#
#
# class BasicBlock111_jingtai_zhuyili(nn.Module):
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
#         super(BasicBlock111_jingtai_zhuyili, self).__init__()
#
#         self.conv3D_3_1 = nn.Conv3d(inplanes, 60, kernel_size=(3, 3, 3), stride=1, padding=1)
#         self.bn1 = nn.BatchNorm3d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_3_2 = nn.Conv3d(60, 120, kernel_size=(3, 3, 3), stride=1, padding=1)
#         self.bn2 = nn.BatchNorm3d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_3_1 = nn.Conv2d(120, 180, kernel_size=(3, 3), stride=1, padding=0)
#         self.bn3 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_3_2 = nn.Conv2d(180, 60, kernel_size=(3, 3), stride=1, padding=0)
#         self.bn4 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_5_1 = nn.Conv3d(30, 60, kernel_size=(3, 3, 3), stride=1, padding=1)
#         self.bn5 = nn.BatchNorm3d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_5_2 = nn.Conv3d(60, 120, kernel_size=(3, 3, 3), stride=1, padding=1)
#         self.bn6 = nn.BatchNorm3d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_5_1 = nn.Conv2d(120, 180, kernel_size=3, stride=1, padding=0)
#         self.bn7 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_5_2 = nn.Conv2d(180, 60, kernel_size=3, stride=1, padding=0)
#         self.bn8 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_7_1 = nn.Conv3d(30, 60, kernel_size=(3, 3, 3), stride=1, padding=1)
#         self.bn9 = nn.BatchNorm3d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_7_2 = nn.Conv3d(60, 120, kernel_size=(3, 3, 3), stride=1, padding=1)
#         self.bn10 = nn.BatchNorm3d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_7_1 = nn.Conv2d(120, 180, kernel_size=(3, 3), stride=1, padding=0)
#         self.bn11 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_7_2 = nn.Conv2d(180, 60, kernel_size=(3, 3), stride=1, padding=0)
#         self.bn12 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.TripletAttention = TripletAttention()
#         self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc3 = nn.Linear(180, 16)
#
#     def forward(self, x):
#         out1 = self.conv3D_3_1(x)
#         out1 = self.bn1(out1)
#         out1 = self.relu(out1)
#
#         out11 = self.conv3D_3_2(out1)
#         out11 = self.bn2(out11)
#         out11 = self.relu(out11)
#
#         out11 = out11.reshape(out11.shape[0], -1, 15, 15)
#         out111 = self.conv2D_3_1(out11)
#         out111 = self.bn3(out111)
#         out111 = self.relu(out111)
#
#         out1111 = self.conv2D_3_2(out111)
#         out1111 = self.bn4(out1111)
#         out1111 = self.relu(out1111)
#         out1111 = self.TripletAttention(out1111)
#         # print("7999",out1111.shape)
#         ###第二层
#
#         out2 = self.conv3D_5_1(x)
#         out2 = self.bn5(out2)
#         out2 = self.relu(out2)
#         # out2 = torch.cat((out1, out2), dim=1)
#
#         out22 = self.conv3D_5_2(out2)
#         out22 = self.bn6(out22)
#         out22 = self.relu(out22)
#
#         out22 = out22.reshape(out22.shape[0], -1, 15, 15)
#
#         out222 = self.conv2D_5_1(out22)
#         out222 = self.bn7(out222)
#         out222 = self.relu(out222)
#         # out222 = torch.cat((out111, out222), dim=1)
#
#         out2222 = self.conv2D_5_2(out222)
#         out2222 = self.bn8(out2222)
#         out2222 = self.relu(out2222)
#         out2222 = self.TripletAttention(out2222)
#         # print("8999", out2222.shape)
#
#         ###第三层
#
#         out3 = self.conv3D_7_1(x)
#         out3 = self.bn9(out3)
#         out3 = self.relu(out3)
#         # out3 = torch.cat((out2, out3), dim=1)
#
#         out33 = self.conv3D_7_2(out3)
#         out33 = self.bn10(out33)
#         out33 = self.relu(out33)
#         out33 = out33.reshape(out33.shape[0], -1, 15, 15)
#
#         out333 = self.conv2D_7_1(out33)
#         out333 = self.bn11(out333)
#         out333 = self.relu(out333)
#         # out333 = torch.cat((out222, out333), dim=1)
#
#         out3333 = self.conv2D_7_2(out333)
#         out3333 = self.bn12(out3333)
#         out3333 = self.relu(out3333)
#         out3333 = self.TripletAttention(out3333)
#         # print("9999", out3333.shape)
#
#         out1 = self.avgpool3(out1111)
#         out4 = self.avgpool3(out2222)
#         out8 = self.avgpool3(out3333)
#
#         out1 = out1.reshape(out1.shape[0], -1)
#         out4 = out4.reshape(out4.shape[0], -1)
#         out8 = out8.reshape(out8.shape[0], -1)
#
#         out = torch.cat((out8, out4, out1), dim=1)
#         out = self.fc3(out)
#
#         return out
#
#
# class BasicBlock111_jingtai_duochidu(nn.Module):
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
#         super(BasicBlock111_jingtai_duochidu, self).__init__()
#
#         self.conv3D_3_1 = nn.Conv3d(inplanes, 60, kernel_size=(3, 3, 3), stride=1, padding=1)
#         self.bn1 = nn.BatchNorm3d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_3_2 = nn.Conv3d(60, 120, kernel_size=(3, 3, 3), stride=1, padding=1)
#         self.bn2 = nn.BatchNorm3d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_3_1 = nn.Conv2d(120, 180, kernel_size=(3, 3), stride=1, padding=0)
#         self.bn3 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_3_2 = nn.Conv2d(180, 60, kernel_size=(3, 3), stride=1, padding=0)
#         self.bn4 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_5_1 = nn.Conv3d(30, 60, kernel_size=(5, 5, 5), stride=1, padding=2)
#         self.bn5 = nn.BatchNorm3d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_5_2 = nn.Conv3d(60, 120, kernel_size=(5, 5, 5), stride=1, padding=2)
#         self.bn6 = nn.BatchNorm3d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_5_1 = nn.Conv2d(120, 180, kernel_size=5, stride=1, padding=0)
#         self.bn7 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_5_2 = nn.Conv2d(180, 60, kernel_size=5, stride=1, padding=0)
#         self.bn8 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_7_1 = nn.Conv3d(30, 60, kernel_size=(7, 7, 7), stride=1, padding=3)
#         self.bn9 = nn.BatchNorm3d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_7_2 = nn.Conv3d(60, 120, kernel_size=(7, 7, 7), stride=1, padding=3)
#         self.bn10 = nn.BatchNorm3d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_7_1 = nn.Conv2d(120, 180, kernel_size=(7, 7), stride=1, padding=0)
#         self.bn11 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_7_2 = nn.Conv2d(180, 60, kernel_size=(7, 7), stride=1, padding=0)
#         self.bn12 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc3 = nn.Linear(180, 16)
#
#     def forward(self, x):
#         out1 = self.conv3D_3_1(x)
#         out1 = self.bn1(out1)
#         out1 = self.relu(out1)
#
#         out11 = self.conv3D_3_2(out1)
#         out11 = self.bn2(out11)
#         out11 = self.relu(out11)
#
#         out11 = out11.reshape(out11.shape[0], -1, 15, 15)
#         out111 = self.conv2D_3_1(out11)
#         out111 = self.bn3(out111)
#         out111 = self.relu(out111)
#
#         out1111 = self.conv2D_3_2(out111)
#         out1111 = self.bn4(out1111)
#         out1111 = self.relu(out1111)
#
#         ###第二层
#
#         out2 = self.conv3D_5_1(x)
#         out2 = self.bn5(out2)
#         out2 = self.relu(out2)
#         # out2 = torch.cat((out1, out2), dim=1)
#
#         out22 = self.conv3D_5_2(out2)
#         out22 = self.bn6(out22)
#         out22 = self.relu(out22)
#
#         out22 = out22.reshape(out22.shape[0], -1, 15, 15)
#
#         out222 = self.conv2D_5_1(out22)
#         out222 = self.bn7(out222)
#         out222 = self.relu(out222)
#         # out222 = torch.cat((out111, out222), dim=1)
#
#         out2222 = self.conv2D_5_2(out222)
#         out2222 = self.bn8(out2222)
#         out2222 = self.relu(out2222)
#
#         ###第三层
#
#         out3 = self.conv3D_7_1(x)
#         out3 = self.bn9(out3)
#         out3 = self.relu(out3)
#         # out3 = torch.cat((out2, out3), dim=1)
#
#         out33 = self.conv3D_7_2(out3)
#         out33 = self.bn10(out33)
#         out33 = self.relu(out33)
#         out33 = out33.reshape(out33.shape[0], -1, 15, 15)
#
#         out333 = self.conv2D_7_1(out33)
#         out333 = self.bn11(out333)
#         out333 = self.relu(out333)
#         # out333 = torch.cat((out222, out333), dim=1)
#
#         out3333 = self.conv2D_7_2(out333)
#         out3333 = self.bn12(out3333)
#         out3333 = self.relu(out3333)
#
#         out1 = self.avgpool3(out1111)
#         out4 = self.avgpool3(out2222)
#         out8 = self.avgpool3(out3333)
#
#         out1 = out1.reshape(out1.shape[0], -1)
#         out4 = out4.reshape(out4.shape[0], -1)
#         out8 = out8.reshape(out8.shape[0], -1)
#
#         out = torch.cat((out8, out4, out1), dim=1)
#         out = self.fc3(out)
#
#         return out
#
#
# def odconv3x3_1(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
#     return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
#                     reduction=reduction, kernel_num=kernel_num)
#
#
# def odconv5x5_1(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
#     return ODConv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2,
#                     reduction=reduction, kernel_num=kernel_num)
#
#
# def odconv7x7_1(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
#     return ODConv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3,
#                     reduction=reduction, kernel_num=kernel_num)
#
#
# class BasicBlock111_dou2D(nn.Module):
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
#         super(BasicBlock111_dou2D, self).__init__()
#
#         self.conv3D_3_1 = odconv3x3_1(inplanes, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn1 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_3_2 = odconv3x3_1(60, 120, reduction=reduction, kernel_num=kernel_num)
#         self.bn2 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_3_1 = odconv3x3(120, 120, reduction=reduction, kernel_num=kernel_num)
#         self.bn3 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_3_2 = odconv3x3(120, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn4 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_5_1 = odconv5x5_1(30, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn5 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_5_2 = odconv5x5_1(60, 120, reduction=reduction, kernel_num=kernel_num)
#         self.bn6 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_5_1 = odconv5x5(120, 120, reduction=reduction, kernel_num=kernel_num)
#         self.bn7 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_5_2 = odconv5x5(120, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn8 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_7_1 = odconv7x7_1(30, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn9 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_7_2 = odconv7x7_1(60, 120, reduction=reduction, kernel_num=kernel_num)
#         self.bn10 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_7_1 = odconv7x7(120, 120, reduction=reduction, kernel_num=kernel_num)
#         self.bn11 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_7_2 = odconv7x7(120, 60, reduction=reduction, kernel_num=kernel_num)
#         self.bn12 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         # self.TripletAttention = TripletAttention()
#         self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc3 = nn.Linear(180, 16)
#
#     def forward(self, x):
#         out1 = self.conv3D_3_1(x)
#         out1 = self.bn1(out1)
#         out1 = self.relu(out1)
#
#         out11 = self.conv3D_3_2(out1)
#         out11 = self.bn2(out11)
#         out11 = self.relu(out11)
#
#         out111 = self.conv2D_3_1(out11)
#         out111 = self.bn3(out111)
#         out111 = self.relu(out111)
#
#         out1111 = self.conv2D_3_2(out111)
#         out1111 = self.bn4(out1111)
#         out1111 = self.relu(out1111)
#         # out1111 = self.TripletAttention(out1111)
#         # print("7999",out1111.shape)
#         ###第二层
#
#         out2 = self.conv3D_5_1(x)
#         out2 = self.bn5(out2)
#         out2 = self.relu(out2)
#         # out2 = torch.cat((out1, out2), dim=1)
#
#         out22 = self.conv3D_5_2(out2)
#         out22 = self.bn6(out22)
#         out22 = self.relu(out22)
#
#         out222 = self.conv2D_5_1(out22)
#         out222 = self.bn7(out222)
#         out222 = self.relu(out222)
#         # out222 = torch.cat((out111, out222), dim=1)
#
#         out2222 = self.conv2D_5_2(out222)
#         out2222 = self.bn8(out2222)
#         out2222 = self.relu(out2222)
#         # out2222 = self.TripletAttention(out2222)
#
#         ###第三层
#
#         out3 = self.conv3D_7_1(x)
#         out3 = self.bn9(out3)
#         out3 = self.relu(out3)
#         # out3 = torch.cat((out2, out3), dim=1)
#
#         out33 = self.conv3D_7_2(out3)
#         out33 = self.bn10(out33)
#         out33 = self.relu(out33)
#
#         out333 = self.conv2D_7_1(out33)
#         out333 = self.bn11(out333)
#         out333 = self.relu(out333)
#         # out333 = torch.cat((out222, out333), dim=1)
#
#         out3333 = self.conv2D_7_2(out333)
#         out3333 = self.bn12(out3333)
#         out3333 = self.relu(out3333)
#         # out3333 = self.TripletAttention(out3333)
#         # print("9999", out3333.shape)
#
#         out1 = self.avgpool3(out1111)
#         out4 = self.avgpool3(out2222)
#         out8 = self.avgpool3(out3333)
#
#         out1 = out1.reshape(out1.shape[0], -1)
#         out4 = out4.reshape(out4.shape[0], -1)
#         out8 = out8.reshape(out8.shape[0], -1)
#
#         out = torch.cat((out8, out4, out1), dim=1)
#         out = self.fc3(out)
#
#         return out
#
#
# class BasicBlock111_jingtai_duochidu_2d(nn.Module):
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
#         super(BasicBlock111_jingtai_duochidu_2d, self).__init__()
#
#         self.conv3D_3_1 = nn.Conv2d(inplanes, 60, kernel_size=(3, 3,), stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_3_2 = nn.Conv2d(60, 120, kernel_size=(3, 3,), stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_3_1 = nn.Conv2d(120, 180, kernel_size=(3, 3), stride=1, padding=0)
#         self.bn3 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_3_2 = nn.Conv2d(180, 60, kernel_size=(3, 3), stride=1, padding=0)
#         self.bn4 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_5_1 = nn.Conv2d(30, 60, kernel_size=(5, 5,), stride=1, padding=2)
#         self.bn5 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_5_2 = nn.Conv2d(60, 120, kernel_size=(5, 5,), stride=1, padding=2)
#         self.bn6 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_5_1 = nn.Conv2d(120, 180, kernel_size=5, stride=1, padding=0)
#         self.bn7 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_5_2 = nn.Conv2d(180, 60, kernel_size=5, stride=1, padding=0)
#         self.bn8 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_7_1 = nn.Conv2d(30, 60, kernel_size=(7, 7), stride=1, padding=3)
#         self.bn9 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_7_2 = nn.Conv2d(60, 120, kernel_size=(7, 7), stride=1, padding=3)
#         self.bn10 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_7_1 = nn.Conv2d(120, 180, kernel_size=(7, 7), stride=1, padding=0)
#         self.bn11 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_7_2 = nn.Conv2d(180, 60, kernel_size=(7, 7), stride=1, padding=0)
#         self.bn12 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc3 = nn.Linear(180, 16)
#
#     def forward(self, x):
#         out1 = self.conv3D_3_1(x)
#         out1 = self.bn1(out1)
#         out1 = self.relu(out1)
#
#         out11 = self.conv3D_3_2(out1)
#         out11 = self.bn2(out11)
#         out11 = self.relu(out11)
#
#         out11 = out11.reshape(out11.shape[0], -1, 15, 15)
#         out111 = self.conv2D_3_1(out11)
#         out111 = self.bn3(out111)
#         out111 = self.relu(out111)
#
#         out1111 = self.conv2D_3_2(out111)
#         out1111 = self.bn4(out1111)
#         out1111 = self.relu(out1111)
#
#         ###第二层
#
#         out2 = self.conv3D_5_1(x)
#         out2 = self.bn5(out2)
#         out2 = self.relu(out2)
#         # out2 = torch.cat((out1, out2), dim=1)
#
#         out22 = self.conv3D_5_2(out2)
#         out22 = self.bn6(out22)
#         out22 = self.relu(out22)
#
#         out22 = out22.reshape(out22.shape[0], -1, 15, 15)
#
#         out222 = self.conv2D_5_1(out22)
#         out222 = self.bn7(out222)
#         out222 = self.relu(out222)
#         # out222 = torch.cat((out111, out222), dim=1)
#
#         out2222 = self.conv2D_5_2(out222)
#         out2222 = self.bn8(out2222)
#         out2222 = self.relu(out2222)
#
#         ###第三层
#
#         out3 = self.conv3D_7_1(x)
#         out3 = self.bn9(out3)
#         out3 = self.relu(out3)
#         # out3 = torch.cat((out2, out3), dim=1)
#
#         out33 = self.conv3D_7_2(out3)
#         out33 = self.bn10(out33)
#         out33 = self.relu(out33)
#         out33 = out33.reshape(out33.shape[0], -1, 15, 15)
#
#         out333 = self.conv2D_7_1(out33)
#         out333 = self.bn11(out333)
#         out333 = self.relu(out333)
#         # out333 = torch.cat((out222, out333), dim=1)
#
#         out3333 = self.conv2D_7_2(out333)
#         out3333 = self.bn12(out3333)
#         out3333 = self.relu(out3333)
#
#         out1 = self.avgpool3(out1111)
#         out4 = self.avgpool3(out2222)
#         out8 = self.avgpool3(out3333)
#
#         out1 = out1.reshape(out1.shape[0], -1)
#         out4 = out4.reshape(out4.shape[0], -1)
#         out8 = out8.reshape(out8.shape[0], -1)
#
#         out = torch.cat((out8, out4, out1), dim=1)
#         out = self.fc3(out)
#
#         return out
#
#
# class BasicBlock111_jingtai_douyou_2d(nn.Module):
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
#         super(BasicBlock111_jingtai_douyou_2d, self).__init__()
#
#         self.conv3D_3_1 = nn.Conv2d(inplanes, 60, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_3_2 = nn.Conv2d(60, 120, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_3_1 = nn.Conv2d(120, 180, kernel_size=3, stride=1, padding=0)
#         self.bn3 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_3_2 = nn.Conv2d(180, 60, kernel_size=3, stride=1, padding=0)
#         self.bn4 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_5_1 = nn.Conv2d(30, 60, kernel_size=5, stride=1, padding=2)
#         self.bn5 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_5_2 = nn.Conv2d(60, 120, kernel_size=5, stride=1, padding=2)
#         self.bn6 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_5_1 = nn.Conv2d(120, 180, kernel_size=5, stride=1, padding=0)
#         self.bn7 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_5_2 = nn.Conv2d(180, 60, kernel_size=5, stride=1, padding=0)
#         self.bn8 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_7_1 = nn.Conv2d(30, 60, kernel_size=7, stride=1, padding=3)
#         self.bn9 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_7_2 = nn.Conv2d(60, 120, kernel_size=7, stride=1, padding=3)
#         self.bn10 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_7_1 = nn.Conv2d(120, 180, kernel_size=7, stride=1, padding=0)
#         self.bn11 = nn.BatchNorm2d(180)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_7_2 = nn.Conv2d(180, 60, kernel_size=7, stride=1, padding=0)
#         self.bn12 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.TripletAttention = TripletAttention()
#         self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc3 = nn.Linear(180, 16)
#
#     def forward(self, x):
#         out1 = self.conv3D_3_1(x)
#         out1 = self.bn1(out1)
#         out1 = self.relu(out1)
#
#         out11 = self.conv3D_3_2(out1)
#         out11 = self.bn2(out11)
#         out11 = self.relu(out11)
#
#         out111 = self.conv2D_3_1(out11)
#         out111 = self.bn3(out111)
#         out111 = self.relu(out111)
#
#         out1111 = self.conv2D_3_2(out111)
#         out1111 = self.bn4(out1111)
#         out1111 = self.relu(out1111)
#         out1111 = self.TripletAttention(out1111)
#         # print("7999",out1111.shape)
#         ###第二层
#
#         out2 = self.conv3D_5_1(x)
#         out2 = self.bn5(out2)
#         out2 = self.relu(out2)
#         # out2 = torch.cat((out1, out2), dim=1)
#
#         out22 = self.conv3D_5_2(out2)
#         out22 = self.bn6(out22)
#         out22 = self.relu(out22)
#
#         out222 = self.conv2D_5_1(out22)
#         out222 = self.bn7(out222)
#         out222 = self.relu(out222)
#         # out222 = torch.cat((out111, out222), dim=1)
#
#         out2222 = self.conv2D_5_2(out222)
#         out2222 = self.bn8(out2222)
#         out2222 = self.relu(out2222)
#         out2222 = self.TripletAttention(out2222)
#         # print("8999", out2222.shape)
#
#         ###第三层
#
#         out3 = self.conv3D_7_1(x)
#         out3 = self.bn9(out3)
#         out3 = self.relu(out3)
#         # out3 = torch.cat((out2, out3), dim=1)
#
#         out33 = self.conv3D_7_2(out3)
#         out33 = self.bn10(out33)
#         out33 = self.relu(out33)
#
#         out333 = self.conv2D_7_1(out33)
#         out333 = self.bn11(out333)
#         out333 = self.relu(out333)
#         # out333 = torch.cat((out222, out333), dim=1)
#
#         out3333 = self.conv2D_7_2(out333)
#         out3333 = self.bn12(out3333)
#         out3333 = self.relu(out3333)
#         out3333 = self.TripletAttention(out3333)
#         # print("9999", out3333.shape)
#
#         out1 = self.avgpool3(out1111)
#         out4 = self.avgpool3(out2222)
#         out8 = self.avgpool3(out3333)
#
#         out1 = out1.reshape(out1.shape[0], -1)
#         out4 = out4.reshape(out4.shape[0], -1)
#         out8 = out8.reshape(out8.shape[0], -1)
#
#         out = torch.cat((out8, out4, out1), dim=1)
#         out = self.fc3(out)
#
#         return out


# from DYConv_2d_GAI import Dynamic_conv2d_gai
#
# class DYConv_2d_GAI_model(nn.Module):
#
#     def __init__(self, inplanes, planes):
#         super(DYConv_2d_GAI_model, self).__init__()
#         # (in_planes=30, out_planes=30, kernel_size=5, ratio=0.25, padding=0,)
#         self.conv3D_3_1 = Dynamic_conv2d_gai(inplanes, 60, kernel_size=3, ratio=0.25, padding=1)
#         self.bn1 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_3_2 = Dynamic_conv2d_gai(60, 120, kernel_size=3,  ratio=0.25, padding=1)
#         self.bn2 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_3_1 = Dynamic_conv2d_gai(120, 120, kernel_size=3, ratio=0.25, padding=0)
#         self.bn3 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_3_2 = Dynamic_conv2d_gai(120, 60, kernel_size=3,  ratio=0.25, padding=0)
#         self.bn4 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_5_1 = Dynamic_conv2d_gai(30, 60,kernel_size=5, ratio=0.25, padding=2)
#         self.bn5 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_5_2 = Dynamic_conv2d_gai(60, 120, kernel_size=5, ratio=0.25, padding=2)
#         self.bn6 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_5_1 = Dynamic_conv2d_gai(120, 120, kernel_size=5, ratio=0.25, padding=0)
#         self.bn7 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_5_2 = Dynamic_conv2d_gai(120, 60, kernel_size=5, ratio=0.25, padding=0)
#         self.bn8 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3D_7_1 = Dynamic_conv2d_gai(30, 60, kernel_size=7, ratio=0.25, padding=3)
#         self.bn9 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3D_7_2 = Dynamic_conv2d_gai(60, 120, kernel_size=7, ratio=0.25, padding=3)
#         self.bn10 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2D_7_1 = Dynamic_conv2d_gai(120, 120, kernel_size=7, ratio=0.25, padding=0)
#         self.bn11 = nn.BatchNorm2d(120)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2D_7_2 = Dynamic_conv2d_gai(120, 60, kernel_size=7, ratio=0.25, padding=0)
#         self.bn12 = nn.BatchNorm2d(60)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.TripletAttention = TripletAttention()
#         self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc3 = nn.Linear(180, 16)
#
#
#
#     def forward(self, x):
#         out1 = self.conv3D_3_1(x)
#         out1 = self.bn1(out1)
#         out1 = self.relu(out1)
#
#         out11 = self.conv3D_3_2(out1)
#         out11 = self.bn2(out11)
#         out11 = self.relu(out11)
#
#         out111 = self.conv2D_3_1(out11)
#         out111 = self.bn3(out111)
#         out111 = self.relu(out111)
#
#         out1111 = self.conv2D_3_2(out111)
#         out1111 = self.bn4(out1111)
#         out1111 = self.relu(out1111)
#         out1111 = self.TripletAttention(out1111)
#
# ###第二层
#
#         out2 = self.conv3D_5_1(x)
#         out2 = self.bn5(out2)
#         out2 = self.relu(out2)
#
#         out22 = self.conv3D_5_2(out2)
#         out22 = self.bn6(out22)
#         out22 = self.relu(out22)
#
#
#         out222 = self.conv2D_5_1(out22)
#         out222 = self.bn7(out222)
#         out222 = self.relu(out222)
#
#
#         out2222 = self.conv2D_5_2(out222)
#         out2222 = self.bn8(out2222)
#         out2222 = self.relu(out2222)
#         out2222 = self.TripletAttention(out2222)
#
#
# ###第三层
#
#         out3 = self.conv3D_7_1(x)
#         out3 = self.bn9(out3)
#         out3 = self.relu(out3)
#
#
#         out33 = self.conv3D_7_2(out3)
#         out33 = self.bn10(out33)
#         out33 = self.relu(out33)
#
#
#
#         out333 = self.conv2D_7_1(out33)
#         out333 = self.bn11(out333)
#         out333 = self.relu(out333)
#
#
#         out3333 = self.conv2D_7_2(out333)
#         out3333 = self.bn12(out3333)
#         out3333 = self.relu(out3333)
#         out3333 = self.TripletAttention(out3333)
#
#
#
#         out1 = self.avgpool3(out1111)
#         out4 = self.avgpool3(out2222)
#         out8 = self.avgpool3(out3333)
#
#         out1 = out1.reshape(out1.shape[0], -1)
#         out4 = out4.reshape(out4.shape[0], -1)
#         out8 = out8.reshape(out8.shape[0], -1)
#
#         out = torch.cat((out8, out4, out1), dim=1)
#         out = self.fc3(out)
#
#         return out
#


# class attention2d(nn.Module):
#     def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
#         super(attention2d, self).__init__()
#         assert temperature % 3 == 1
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         if in_planes != 3:
#             hidden_planes = int(in_planes * ratios) + 1
#         else:
#             hidden_planes = K
#         self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
#         # self.bn = nn.BatchNorm2d(hidden_planes)
#         self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
#         self.temperature = temperature
#         if init_weight:
#             self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             if isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def updata_temperature(self):
#         if self.temperature != 1:
#             self.temperature -= 3
#             print('Change temperature to:', str(self.temperature))
#
#     def forward(self, x):
#         x = self.avgpool(x)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x).view(x.size(0), -1)
#         return F.softmax(x / self.temperature, 1)
#
#
# class Dynamic_conv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1,
#                  bias=True, K=4, temperature=34, init_weight=True):
#         super(Dynamic_conv2d, self).__init__()
#         assert in_planes % groups == 0
#         self.in_planes = in_planes
#         self.out_planes = out_planes
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.bias = bias
#         self.K = K
#         self.attention = attention2d(in_planes, ratio, K, temperature)
#
#         self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size),
#                                    requires_grad=True)
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(K, out_planes))
#         else:
#             self.bias = None
#         if init_weight:
#             self._initialize_weights()
#
#         # TODO 初始化
#
#     def _initialize_weights(self):
#         for i in range(self.K):
#             nn.init.kaiming_uniform_(self.weight[i])
#
#     def update_temperature(self):
#         self.attention.updata_temperature()
#
#     def forward(self, x):  # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
#         softmax_attention = self.attention(x)
#         batch_size, in_planes, height, width = x.size()
#         x = x.view(1, -1, height, width)  # 变化成一个维度进行组卷积
#         weight = self.weight.view(self.K, -1)
#
#         # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
#         aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size * self.out_planes,
#                                                                     self.in_planes // self.groups, self.kernel_size,
#                                                                     self.kernel_size)
#         if self.bias is not None:
#             aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
#             output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
#                               dilation=self.dilation, groups=self.groups * batch_size)
#         else:
#             output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
#                               dilation=self.dilation, groups=self.groups * batch_size)
#
#         output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
#         return output

#
class DYConv_2d(nn.Module):

    def __init__(self, inplanes, planes):
        super(DYConv_2d, self).__init__()
        # (in_planes=30, out_planes=30, kernel_size=5, ratio=0.25, padding=0,)
        self.conv3D_3_1 = Dynamic_conv2d(inplanes, 60, kernel_size=3, ratio=0.25, padding=1)
        self.bn1 = nn.BatchNorm2d(60)
        self.relu = nn.ReLU(inplace=True)
        self.conv3D_3_2 = Dynamic_conv2d(60, 120, kernel_size=3, ratio=0.25, padding=1)
        self.bn2 = nn.BatchNorm2d(120)
        self.relu = nn.ReLU(inplace=True)

        self.conv2D_3_1 = Dynamic_conv2d(120, 120, kernel_size=3, ratio=0.25, padding=0)
        self.bn3 = nn.BatchNorm2d(120)
        self.relu = nn.ReLU(inplace=True)
        self.conv2D_3_2 = Dynamic_conv2d(120, 64, kernel_size=3, ratio=0.25, padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)


        self.TripletAttention = TripletAttention()
        self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool2 = nn.MaxPool2d((9, 9))
        self.fc3 = nn.Linear(100, 64)

    def forward(self, x):


        x1 = self.avgpool3(x)
        # print(x1.shape)
        x1 = x1.reshape(x1.shape[0], -1)
        x1 = self.fc3(x1)




        out1 = self.conv3D_3_1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out11 = self.conv3D_3_2(out1)
        out11 = self.bn2(out11)
        out11 = self.relu(out11)

        out111 = self.conv2D_3_1(out11)
        out111 = self.bn3(out111)
        out111 = self.relu(out111)

        out1111 = self.conv2D_3_2(out111)
        out1111 = self.bn4(out1111)
        out1111 = self.relu(out1111)
        out1111 = self.TripletAttention(out1111)

        out1 = self.avgpool3(out1111)

        out1 = out1.reshape(out1.shape[0], -1)

        out = []

        out.append(x1)
        out.append(out1)



        return out
