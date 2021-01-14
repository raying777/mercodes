# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:22:32 2020

@author: ray
修改记录：46，49 行是版本更新后的警告，改过之后就好了。
"""

import torch
#import torchvision.models as models
from mymodel import TCN
from mydataset import MyData
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
print('the test in loso(26 in total) of casme2(5 class) :')
batch_size = 64
input_channels = 1 
seq_length = int(1372 / input_channels)
n_classes = 5
total_correct = []
total_test_number = []
total_acc2 = 0
for index in range(1,27):
    
    test_set = MyData(mat_root=r'.\data\mat\feature_5000_gai.mat', 
                           data_root = r'.\data\test\test'+str(index)+'.txt',
                           transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    #读数据
    model = TCN(input_channels, n_classes, [25,25,25,25], [25], 7, 0.05)
    net = torch.load(r'.\data\model\newbest'+str(index)+'.pth')
    model.load_state_dict(net)
    #print(net)
    model.cuda()
    model.eval()
    test_loss = 0
    correct = 0
#    if index == 0: 
#        print(model)
    
    for data, target in test_loader:
    
        data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)
        #data, target = Variable(data, volatile=True), Variable(target)
        data, target = Variable(data), Variable(target)
        output = model(data)
        #test_loss += F.nll_loss(output, target, size_average=False).item()
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        target_npy = target.cpu().numpy()
        target_list = target_npy.tolist()
        pred_npy = pred.cpu().numpy().reshape(len(target_list))
        pred_list = pred_npy.tolist()
        total_correct.append(int(correct))
    test_loss /= len(test_loader.dataset)
    total_test_number.append(len(test_loader.dataset))
    print('the {}th test:  test acc:{}%   test loss:{}'.format(index, 100. * correct / len(test_loader.dataset), test_loss))    

total_test_samples = sum(total_test_number)
total_correct_samples = sum(total_correct)
total_acc =  total_correct_samples/total_test_samples

print('total acc: {}'.format(total_acc))
'''
#another acc 
for i in range(0,26):
    total_acc1 = total_correct[i]/total_test_number[i]
    total_acc2+= total_acc1
total_acc2 = total_acc2 / 26 

print('total acc: {}'.format(total_acc))
print('another total acc: {}'.format(total_acc2))
'''