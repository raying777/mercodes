# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:35:52 2019

@author: lei
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    #view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，
    #需要用contiguous()来返回一个contiguous copy。 


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        #n_inputs=1,n_outputs=25,kernel_size=2,stride=1,dilation=1、2、4、8,padding=1、2、4、8,dropout=0.2
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        #Sequential是一个有序容器神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，
        #同时以神经网络模块为元素的有序字典也可以作为传入参数。
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        #xxx.weight.data是卷积核参数
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels_p, num_channels_s, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()#继承父类的
        layers_p = []
        layers_s = []

        num_levels_p = len(num_channels_p)
        num_levels_s = len(num_channels_s)
        for i in range(num_levels_p):
            dilation_size_p = 2 ** i
            in_channels_p = num_inputs if i == 0 else num_channels_p[i-1]#num_inputs=1,in_channels = 1
            out_channels_p = num_channels_p[i]#25
            layers_p += [TemporalBlock(in_channels_p, out_channels_p, kernel_size, stride=1, dilation=dilation_size_p,
                                     padding=(kernel_size-1) * dilation_size_p, dropout=dropout)]
        for j in range(num_levels_s):
            dilation_size_s = 49
            in_channels_s = num_inputs if j == 0 else num_channels_s[j-1]#num_inputs=1,in_channels = 1
            out_channels_s = num_channels_s[j]#25
            layers_s += [TemporalBlock(in_channels_s, out_channels_s, kernel_size, stride=1, dilation=dilation_size_s,
                                     padding=(kernel_size-1) * dilation_size_s, dropout=dropout)]
        self.network_p = nn.Sequential(*layers_p)
        self.network_s = nn.Sequential(*layers_s)
    def forward(self, x):
        return self.network_p(x)+self.network_s(x)
