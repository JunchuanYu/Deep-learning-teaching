
# coding: utf-8

# # UNET模型

# [学习链接](https://blog.csdn.net/weixin_44791964/article/details/108866828)
# # 数据读取  模型建立、训练、保存 

# In[21]:



import os,gc
#os.environ['CUDA_VISIBLE_DEVICES']="0,1"
import cv2
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision


transform = transforms.Compose([
    transforms.ToTensor()
])



  
#############################            UNet        ####################
#       下采样
class DownBlock(nn.Module):
    def __init__(self, num_convs, inchannels, outchannels, pool=True):
        super(DownBlock, self).__init__()
        blk = []
        if pool:
            blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for i in range(num_convs):
            if i == 0:
                blk.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1))
            else:
                blk.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1))
            blk.append(nn.ReLU(inplace=True))
        self.layer = nn.Sequential(*blk)

    def forward(self, x):
        return self.layer(x)
    
#     上采样
class UpBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(UpBlock, self).__init__()
        self.convt = nn.ConvTranspose2d(inchannels, outchannels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.convt(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    
    
class UNet_1(nn.Module):
    def __init__(self, nchannels=1, nclasses=1):
        super().__init__()
        self.down1 = DownBlock(2, nchannels, 64, pool=False)
        self.down2 = DownBlock(3, 64, 128)
        self.down3 = DownBlock(3, 128, 256)
        self.down4 = DownBlock(3, 256, 512)
        self.down5 = DownBlock(3, 512, 1024)
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.out = nn.Sequential(
            nn.Conv2d(64, nclasses, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)

    
 ###############################################################################



class Conv_Block(nn.Module):       #卷积块，包括两层卷积，两个RELU，两个BN
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),   #若使用BN，则不要有bias 
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
        
    def forward(self,x):
        return self.layer(x)


class DownSample(nn.Module):         #下采样
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            #nn.MaxPool2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),  #Z最大池化可能会丢失信息
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)


class UpSample(nn.Module):       # 上采样
    def __init__(self,channel):
        super(UpSample, self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)       #通道压缩
    def forward(self,x,feature_map):                        
        up=F.interpolate(x,scale_factor=2,mode='nearest')  #使用最邻近插值法，扩大两倍
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)          #与前面得特征层做堆叠（N,C,H,W）


class UNet___________(nn.Module):
    def __init__(self,num_classes):
        super(UNet, self).__init__()
        self.c1=Conv_Block(6,64)
        self.d1=DownSample(64)
        self.c2=Conv_Block(64,128)
        self.d2=DownSample(128)
        self.c3=Conv_Block(128,256)
        self.d3=DownSample(256)
        self.c4=Conv_Block(256,512)
        self.d4=DownSample(512)
        self.c5=Conv_Block(512,1024)
        self.u1=UpSample(1024)
        self.c6=Conv_Block(1024,512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.out=nn.Conv2d(64,num_classes,3,1,1)

    def forward(self,x):
        #主干特征提取
        R1=self.c1(x)
        R2=self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        
        #加强特征提取
        O1=self.c6(self.u1(R5,R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.out(O4)

