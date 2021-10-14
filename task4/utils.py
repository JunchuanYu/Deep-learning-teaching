import cv2
import numpy as np
import  os,sys
from osgeo import gdal
from gdalconst import *
import matplotlib.pyplot as plt
from PIL import Image
import sys, time  
import matplotlib
import scipy.misc
import glob
import shutil
import h5py
from random import shuffle
from keras.utils.np_utils import to_categorical

def Load_image_by_Gdal(file_path):
    img_file = gdal.Open(file_path, GA_ReadOnly)
    img_bands = img_file.RasterCount#band num
    img_height = img_file.RasterYSize#height
    img_width = img_file.RasterXSize#width
    img_arr = img_file.ReadAsArray()#获取投影信息
    geomatrix = img_file.GetGeoTransform()#获取仿射矩阵信息
    projection = img_file.GetProjectionRef()
    return img_file, img_bands, img_height, img_width, img_arr, geomatrix, projection
def Write_Tiff(img_arr, geomatrix, projection,path):
#     img_bands, img_height, img_width = img_arr.shape
    if 'int8' in img_arr.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img_arr.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(img_arr.shape) == 3:
        img_bands, img_height, img_width = img_arr.shape
    elif len(img_arr.shape) == 2:
        img_arr = np.array([img_arr])
        img_bands, img_height, img_width = img_arr.shape
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(img_width), int(img_height), int(img_bands), datatype)
    print(path, int(img_width), int(img_height), int(img_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(geomatrix) #写入仿射变换参数
        dataset.SetProjection(projection) #写入投影
    for i in range(img_bands):
        dataset.GetRasterBand(i+1).WriteArray(img_arr[i])
    del dataset
def read_tiff(file):
    img_bands, img_height, img_width, img_arr=Load_image_by_Gdal(file)
    img_arr=img_arr.transpose(( 1, 2,0))
    return img_arr
def rotate(arr_image):
    rotate90 = np.rot90(arr_image)
    rotate180 = np.rot90(rotate90)
    rotate270 = np.rot90(rotate180)
    print(rotate90.shape,rotate180.shape,rotate270.shape)
    arr_90 = rotate90.transpose(1,0,2,3)
    arr_180 = rotate180.transpose(0,2,1,3)
    arr_270 = rotate270.transpose(1,0,2,3)
    arr_image = np.concatenate((arr_image, arr_90), axis = 0)
    arr_image = np.concatenate((arr_image, arr_180), axis = 0)
#     arr_image = np.concatenate((arr_image, arr_270), axis = 0)
    print(arr_90.shape,arr_180.shape,arr_270.shape)
    return arr_image
def suffle_data(imgd):
    index = [i for i in range(len(imgd))]
    shuffle(index)
    shuffle(index)
    shuffle(index)
    newimg = imgd[index, :, :, :]
    print(newimg.shape)
    return newimg
def plot_func_20(trainx,trainy):
    fig=plt.figure(figsize=(25,5))
    for i in range(20):   
        plt.subplot(2,10,i+1)
        plt.imshow(trainx[i,:,:,:])
    plt.show()
    fig=plt.figure(figsize=(25,5))
    for i in range(20):   
        plt.subplot(2,10,i+1)
        plt.imshow(trainy[i,:,:],cmap="gray")
    plt.show()
def regular_crop(imagearray, crop_sz,step=512):
    data_all = []
    for i in np.arange(1):
        data = []
        x=0
#         src_img_row,src_img_col=(imagearray.size)[0:2]
        row_num = ((imagearray.shape)[0] - step) // step  ###最多能裁剪几行 几列
        col_num=((imagearray.shape)[1] - step) // step
    #print(row_num,col_num)
        x_start=0
        y_start=0
        for h in range(row_num):
            for w in range(col_num):
                crop_img = imagearray[crop_sz*h+y_start:crop_sz*(h+1)+y_start, crop_sz*w+x_start:crop_sz*(w+1)+x_start,:]               
#                 crop_img= imagearray.crop((step * h, step * w,step * h + crop_sz,step * w + crop_sz))
                data.append(crop_img)
                x=x+1
                if x % 10 ==0:
                    print("processing....patch:"+str(i)+"...No.:"+str(x))
        data=np.array(data)
        print("processing....patch:"+str(i)+"..Total.No.:"+str(data.shape))
        if i == 0:
            data_all = data
        else:
            data_all=np.concatenate((data_all, data), axis = 0)
    return data_all
def plot_func(data,label):
    fig=plt.figure(figsize=(25,5))
    for i in range(20):
        plt.subplot(2,10,i+1)
        plt.imshow(Image.fromarray(np.uint8((data[i,:,:,0:3])*255)))
    plt.show()
    fig=plt.figure(figsize=(25,5))
    for i in range(20):
        plt.subplot(2,10,i+1)
        plt.imshow((label[i,:,:,0]),cmap="gray")
    plt.show()
def label_hot(label,n_label=1):
    listlabel=[]
    for i in label:
        mask=i.flatten()
        mask=to_categorical(mask, num_classes=n_label)
        listlabel.append(mask)
    msk=np.asarray(listlabel,dtype='float32')
    msk=msk.reshape((label.shape[0],label.shape[1],label.shape[2],n_label))
#     print(msk.shape)
    return msk
def data_split(data,sband,overdata1,overdata2):
    data1=data[:,:,:,:sband]
    data2=data[:,:,:,sband:]
    data1=percent_remove(data1,overdata1)
    data1=np.array(data1,dtype='float32')/overdata1
    # print(np.max(data1))
    data2=percent_remove(data2,overdata2)
    data2=np.array(data2,dtype='float32')/overdata2
    # print(np.max(data2),np.min(data2))
    data=np.concatenate((data1,data2), axis=-1)
    return data
def percent_remove(data,overdata,min_per=0.01,max_per=99.99):
    min=np.percentile(data,min_per)
    max=np.percentile(data,max_per)
    data[data<min]=min
    data[data>max]=max
    print(min,max)
    data[np.where(data>overdata)]=overdata
    return data
def get_normalized_patches(image,label,sband,overdata1,overdata2,n_label=1):
#     data = get_all_patches()
#     data = np.load(Dir + '\output\data_pos_%d_%d_class%d.npy' % (Patch_size, N_split, Class_Type))
    # img=percent_remove(img,overdata1)
    image=data_split(image,sband,overdata1,overdata2)
    # img[img>1200]=1200
    # img=np.array(image,dtype='float32')/overdata1
    # img=np.asarray(img,dtype='uint16')
    # label=label//255.0
    msk = label_hot(label,n_label)
    # print(img.shape,msk.shape)
    # print(img.dtype)
    #     combin=np.expand_dims(combin, axis=3)
#     img=np.concatenate((img,combin),axis=-1)
    return image,msk
def val_plot_func(data,label,yval):
    fig=plt.figure(figsize=(25,5))
    for i in range(20):   
        plt.subplot(2,10,i+1)
        plt.imshow(Image.fromarray(np.uint8((data[i,:,:,0:3])*255)))
    plt.show()
    fig=plt.figure(figsize=(25,5))
    for i in range(20):   
        plt.subplot(2,10,i+1)
        plt.imshow((label[i,:,:]))
    plt.show()
    fig=plt.figure(figsize=(25,5))
    for i in range(20):   
        plt.subplot(2,10,i+1)
        plt.imshow((yval[i,:,:]))
    plt.show()
def predict(model,image,ProductDir,shotname,threshold,n_label,is_score=1):
    #print("[INFO] loading network...")
    stride=128   #128
    image_size=256
    h,w,c = image.shape
    padding_h = (h//stride + 1) * stride
    padding_w = (w//stride + 1) * stride
    padding_img = np.zeros((padding_h,padding_w,c),dtype=np.float32)
    padding_img[0:h,0:w,:] = image[:,:,:]
    padding_img = padding_img.astype("float")
    # padding_img = img_to_array(padding_img)
    # 	print ('src:',padding_img.shape)
    mask_whole = np.zeros((padding_h,padding_w,n_label),dtype=np.float32)
    print('all images='+str((padding_h//stride)*(padding_w//stride)))
    for i in range(padding_h//stride):
        if i % 10 ==0 :
            print('processing.....' +str(i)+'    of   '+str(padding_h//stride))
        for j in range(padding_w//stride):
            crop = padding_img[i*stride:i*stride+image_size,j*stride:j*stride+image_size,:]
            #         print(crop.shape)
            ch,cw,c = crop.shape
            if ch != image_size or cw != image_size:
            #             print ('invalid size!')
                continue
            crop = np.expand_dims(crop, axis=0)
            pred = model.predict(crop,verbose=0)
#             print(pred.shape)
            #pred=np.argmax(pred, axis=-1)
            #print (np.unique(pred))
            pred = pred.reshape((image_size,image_size,n_label)).astype(np.float32)
#             print(pred.shape)
            #print 'pred:',pred.shape
            mask_whole[i*stride:i*stride+image_size,j*stride:j*stride+image_size,:] = pred[:,:,:]
            #mask_whole[i * stride:i * stride + image_size , j * stride:j * stride + image_size,:] = pred[:,:,:]

    img_final = mask_whole[0:h, 0:w,:]
#    print(np.mean(img_final))
    # imgarr=np.zeros((h,w),dtype=np.uint8)

#    if is_score==1:
#        for k in range(n_label):
#            im=np.array((img_final[:,:,i]*256),dtype=np.uint8)
#            imwrite((ProductDir+shotname+'_score'+str(k)+'.png'),im)
#            imgarr[img_final[:,:,i]>threshold]=1
#            imgarr[img_final[:,:,i]<=threshold]=0
#            im=np.array(imgarr*255,dtype=np.uint8)   
#            cv2.imwrite((ProductDir+shotname+'_pred'+str(k)+'.png'),im)

    #img_final[img_final>=1]=1
    pred=np.argmax(img_final, axis=-1)
    # cv2.imwrite((ProductDir+shotname+'_prdlabels.png'),pred*255)
    #print (np.unique(img_final))
    return pred,img_final

def out_report(gt_arr,label_arr,out_csv):
    # gt_arr=Load_image_by_Gdal(gt)
    gt_reshape=gt_arr.reshape((gt_arr.shape[0]*gt_arr.shape[1],1))
    label_reshape=label_arr.reshape((label_arr.shape[0]*label_arr.shape[1],1))
    acc=classification_report(gt_reshape,label_reshape,output_dict=True)
    df=pd.DataFrame(acc).transpose()
    df.to_csv(out_csv,index=True)   
def value_set(arr,labels):
    n=len(labels)
    for i in range(n):
        if i ==0:
            arr[i,:,:][np.where(arr[i,:,:] !=0)]==labels[0]
        else:
            arr[i,:,:][np.where(arr[i,:,:]==1)]=labels[i]
    return arr
def threshold_arr(arr,threshold=0.5):
    # arr=load_tif(file)
    result=np.zeros(arr.shape)
    for i in range(arr.shape[2]):
        result[:,:,i][np.where(arr[:,:,i]>threshold)]=1
    result=np.array(result,dtype=np.uint8)
    # result=(result.transpose(( 2,0,1)))
    print(result.shape,np.max(result))
    return result
def argmax_out(arr,geomatrix,projection,outfile,threshold=0.5):
    arr=threshold_arr(arr,threshold)
    result=arr
    # result=value_set(result,labels)
    # (filepath,filename) = os.path.split(file)
    # (shotname,extension) = os.path.splitext(filename)
    finalresult=np.argmax(result,axis=0)
    np.max(finalresult)
    # outfile=filepath+os.path.sep+shotname+'_label.tif'
    Write_Tiff(finalresult, geomatrix, projection,outfile)
def plot_label_func(data,label,n_label):
    fig=plt.figure(figsize=(25,5))
    for i in range(n_label):
        plt.subplot(1,n_label,i+1)
        plt.imshow((data[:,:,i]))
    plt.show()
    fig=plt.figure(figsize=(25,5))
    for i in range(n_label):
        plt.subplot(1,n_label,i+1)
        plt.imshow((label[:,:,i]))
    plt.show()