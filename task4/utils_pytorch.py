#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os,gc
import cv2
import h5py
import tqdm
import torch
import numpy as np
from osgeo import gdal
import glob
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
from torchsummary import summary
from sklearn.metrics import accuracy_score

transform = transforms.Compose([
    transforms.ToTensor()
])


# In[ ]:


class Dataset_hdf5(Dataset):
    def __init__(self, path, num_classes):
        fd = h5py.File(path)
        fd.keys()
        images,labels=np.asarray(fd['image']),np.asarray(fd['label'])
        fd.close()
        img = np.transpose(images.astype(np.float32), (0,3,1,2)) #转为NCHW
        images =  data_split(img,3,10000,32000)
        labels = labels/1.0
        lables = torch.FloatTensor(labels)
        self.labels = labels
        self.images = images
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
        
    def __len__(self):
        return self.labels.shape[0]

    
def train_test_dataset(full_dataset, p):
    assert p > 0 and p < 1  
    train_size = int(p * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset





def label_hot(label,n_label=1):
    listlabel=[]
    for i in label:
        mask=i.flatten()
        mask = mask.astype(np.int64)
        mask=F.one_hot(torch.as_tensor(mask), num_classes=n_label)
        
        listlabel.append(mask.numpy())
    msk=np.asarray(listlabel,dtype='float32')
    msk=msk.reshape(label.shape[0],label.shape[1],label.shape[2],n_label)
    msk = np.transpose(msk, (0,3,1,2))
    return msk

def Load_image_by_Gdal(file_path):
    img_file = gdal.Open(file_path, gdal.GA_ReadOnly)
    img_bands = img_file.RasterCount#band num
    img_height = img_file.RasterYSize#height
    img_width = img_file.RasterXSize#width
    img_arr = img_file.ReadAsArray()#获取投影信息
    geomatrix = img_file.GetGeoTransform()#获取仿射矩阵信息
    projection = img_file.GetProjectionRef()
    return img_file, img_bands, img_height, img_width, img_arr, geomatrix, projection

def data_split(data,sband,overdata1,overdata2):
    data1=data[:,:sband,:,:]
    data2=data[:,sband:,:,:]
    data1=percent_remove(data1,overdata1)
    data1=np.array(data1,dtype='float32')/overdata1
    # print(np.max(data1))
    data2=percent_remove(data2,overdata2)
    data2=np.array(data2,dtype='float32')/overdata2
    # print(np.max(data2),np.min(data2))
    data=np.concatenate((data1,data2), axis=1)
    return data

def percent_remove(data,overdata,min_per=0.01,max_per=99.99):
    min=np.percentile(data,min_per)
    max=np.percentile(data,max_per)
    data[data<min]=min
    data[data>max]=max
    print(min,max)
    data[np.where(data>overdata)]=overdata
    return data

def acc_fn(per, label):
    n,_,_,_ =per.shape
    per = torch.sigmoid(per)
    per = torch.argmax(per,dim=1)
    per = torch.flatten(per, start_dim=1)
    label = torch.flatten(label, start_dim=1)
    train_acc = 0
    for i in range(n):
        acc  = accuracy_score(label[i,:],per[i,:])
        train_acc += acc
    return train_acc/n
