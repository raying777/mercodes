# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:35:33 2019

@author: lei
"""

import torch.nn.functional as F
from torch import nn
from mytcn import TemporalConvNet
#from mygraphtcn import TemporalConvNet
import torch

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels_p, num_channels_s, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels_p, num_channels_s, kernel_size=kernel_size, dropout=dropout)
        self.linear1 = nn.Linear(num_channels_p[-1]*1372, 512)
        self.linear2 = nn.Linear(512, output_size)
        self.relu3 = nn.ReLU()
        #self.relu4 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.bn0 = nn.BatchNorm1d(num_channels_p[-1]*1372)
        self.mp1 = nn.MaxPool1d(1000)
    def forward(self, inputs):
        #把数据分为两个部分，一个是眉毛，一个是嘴巴。
        #print(inputs.shape)
        eye_inputs = inputs[:,:,0:490]
        mou_inputs = inputs[:,:,490:1372]
        #print(eye_inputs.shape)        
        #print(mou_inputs.shape)
        #print(inputs.shape)
        """Inputs have to have dimension (N, C_in, L_in)"""
        ye = self.tcn(eye_inputs)  # input should have dimension (N, C, L)
        ym = self.tcn(mou_inputs)
        #print(yp.shape)        
        #print(ys.shape)
        y = torch.cat([ye, ym], dim=2)
        #print(y.shape)
        y1 = y.view(y.size()[0], -1)
        #o0 = self.mp1(y)
       # y1 = y.view(o0.size()[0], -1)
        #o0 = self.bn0(y1)
        #print(y1.shape)
        #print(y.shape)
        #o = self.linear1(y1)
        #o1 = self.relu3(o)
        #o2 = self.dropout3(o)
        #print(o.shape)
        #o1 = self.bn1(o)
        #print(o1.shape)
        #o3 = self.linear2(o1)
        #o4 = self.bn2(o3)
        #o4 = self.dropout4(o3)
        
        #print(o2.shape)
        #o3 = self.relu4(o2)
        #print(o3.shape)
        o0 = self.relu3(self.dropout3(self.bn1(self.linear1(y1))))
        o1 = self.dropout4(self.bn2(self.linear2(o0)))
        
        return F.log_softmax(o1, dim=1)
