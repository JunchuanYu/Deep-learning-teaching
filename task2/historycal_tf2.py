import matplotlib.pyplot as plt
import h5py
from tensorflow.keras.callbacks import Callback
import numpy as np

class LossHistory(Callback):
    '''
    tf2 版本的acc改为accuracy，val_acc改为val_accuracy
    在模型开始的时候定义四个属性，每一个属性都是字典类型，存储相对应的值和epoch
    '''
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    # 在每一个batch结束后记录相应的值
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    # 在每一个epoch之后记录相应的值
    def on_epoch_end(self, epoch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))
        
    def loss_plot(self, loss_type,nettag):
        '''
        loss_type：指的是 'epoch'或者是'batch'，分别表示是一个batch之后记录还是一个epoch之后记录
        '''
        if loss_type == 'batch':
            iters = len(self.losses[loss_type])
            plt.style.use("ggplot")
            plt.figure(figsize=(10,6))
            plt.plot(np.arange(0, iters), self.losses[loss_type],label='Train_Loss')
#             plt.plot(np.arange(0, iters), self.val_loss[loss_type],label='Val_Loss')
            plt.plot(np.arange(0, iters), self.accuracy[loss_type],label='Train_Acc')
#             plt.plot(np.arange(0, iters), self.val_acc[loss_type],label='Val_Acc')
            plt.ylim(0,1)
            plt.title("Training Loss and Accuracy")
            plt.xlabel("Iteration")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")
            # plt.savefig(nettag+"_Loss_Acc_batch.png")
            plt.show()
        elif loss_type == 'epoch':
            iters = len(self.losses[loss_type])
            plt.style.use("ggplot")
            plt.figure(figsize=(10,6))
            plt.plot(np.arange(0, iters), self.losses[loss_type],label='Train_Loss')
            plt.plot(np.arange(0, iters), self.val_loss[loss_type],label='Val_Loss')
            plt.plot(np.arange(0, iters), self.accuracy[loss_type],label='Train_Acc')
            plt.plot(np.arange(0, iters), self.val_acc[loss_type],label='Val_Acc')
            plt.ylim(0,1)
            plt.title("Training Loss and Accuracy ")
            plt.xlabel("Epoch")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")
            # plt.savefig(nettag+"_Loss_Acc_epoch.png")
            plt.show()   
    def h5_save(self,h5_path):
        f = h5py.File(h5_path, mode='w')
        f['acc_epoch'] = self.accuracy['epoch']
        f['acc_batch'] = self.accuracy['batch']
        f['loss_epoch'] = self.losses['epoch']
        f['loss_batch'] = self.losses['batch']
        f['val_acc'] = self.val_acc['epoch']
        f['val_loss'] = self.val_loss['epoch']
        f.close()