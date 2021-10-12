from tensorflow.keras.callbacks import Callback,CSVLogger,ReduceLROnPlateau,EarlyStopping
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
# from keras.layers import Conv2D,Concatenate,Input,Dropout,ZeroPadding2D,AveragePooling2D,BatchNormalization,Activation,Add,UpSampling2D,MaxPooling2D,Reshape,Lambda
from tensorflow.keras.optimizers import Adam,Adamax,Nadam,Adadelta,SGD,RMSprop
from tensorflow.keras.callbacks import CSVLogger,ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,TensorBoard
from tensorflow.keras.callbacks import Callback,CSVLogger,ReduceLROnPlateau,EarlyStopping
# from tensorflow.keras.engine import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model




class all_model(object):
    def __init__(self,loss,loss_weights,optimizer,metrics,input_height,input_width,nclass,nchannel):
        self.LOSS = loss
        self.OPTIMIZER = optimizer
        self.METRICS = metrics
        self.input_height = input_height
        self.input_width=input_width
        self.nClasses=nclass
        self.nchannel=nchannel
        self.model = None
        self.img_input=Input(shape=(self.input_height, self.input_width, self.nchannel))
        self.loss_weights=loss_weights
  ## VGG16_head

    def VGG16_head(self):
        #block1
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',name='block1_conv1')(self.img_input)
#         x = (BatchNormalization())(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',name='block1_conv2')(x)
#         x = (BatchNormalization())(x)
        x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)
        f1=x #128
        #block2
        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',name='block2_conv1')(x)
#         x = (BatchNormalization())(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',name='block2_conv2')(x)
#         x = (BatchNormalization())(x)
        x = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(x)
        f2=x #64
        #block3
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',name='block3_conv1')(x)
#         x = (BatchNormalization())(x)
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',name='block3_conv2')(x)
#         x = (BatchNormalization())(x)
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',name='block3_conv3')(x)
#         x = (BatchNormalization())(x)
        x = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(x)
        f3=x #32
        #block4
        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu',name='block4_conv1')(x)
#         x = (BatchNormalization())(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu',name='block4_conv2')(x)
#         x = (BatchNormalization())(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu',name='block4_conv3')(x)
#         x = (BatchNormalization())(x)
        x = MaxPooling2D(pool_size=(2, 2), name='block4_pool')(x)
        f4=x #16
        #block5
        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu',name='block5_conv1')(x)
#         x = (BatchNormalization())(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu',name='block5_conv2')(x)
#         x = (BatchNormalization())(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu',name='block5_conv3')(x)
#         x = (BatchNormalization())(x)
        x = MaxPooling2D(pool_size=(2, 2), name='block5_pool')(x)
        f5=x #8
                
        fc6 = Conv2D(filters=4096, kernel_size=(1, 1), padding='same', activation='relu',name='fc6')(x)
        fc6 = Dropout(0.3, name='dropout_1')(fc6)

        fc7 = Conv2D(filters=4096, kernel_size=(1, 1), padding='same', activation='relu',name='fc7')(fc6)
        fc7 = Dropout(0.3, name='dropour_2')(fc7)
        
        fc8= Conv2D(filters=self.nClasses, kernel_size=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal", name="score_fr")(fc7)
        ##size conver from 8 to 16
        fc8 = Conv2DTranspose(filters=self.nClasses, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None, name="score2")(fc8)        
#         fcn8 = Model(inputs=img_input, outputs=o)

        return [f1, f2, f3, f4, f5,fc6,fc7,fc8]
   ## RESNET50_head    
    def identity_block(self,input_tensor, kernel_size, filters, stage, block,dilation_rate=1):

        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,padding='same', dilation_rate=dilation,name=conv_name_base + '2b')(x)
        x = BatchNormalization( name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(name=bn_name_base + '2c')(x)

        x = add([x, input_tensor])
        x = Activation('relu')(x)
        return x
    def conv_block(self,input_tensor, kernel_size, filters, stage, block,strides=(2, 2),dilation_rate=1):
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,padding='same',dilation_rate=dilation, name=conv_name_base + '2b')(x)
        x = BatchNormalization(name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

        x = add([x, shortcut])
        x = Activation('relu')(x)
        return x
    def RESNET50_head(self):
#         x = ZeroPadding2D((3, 3))(self.img_input)
        input_shape=self.img_input
        x = Conv2D(64, (3,3), strides=(2, 2), name='conv1',padding='same')(input_shape)
        f1 = x

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3),strides=(2, 2),padding='same')(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')
        f2 =x

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')
        f3 = x
        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
        f4 = x

        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
        f5 = x
#         x = AveragePooling2D((7, 7), name='avg_pool')(x)
        # f6 = x
        return input_shape,[f1, f2, f3, f4, f5]   

   ## SQUEESE_head
    def SqueezeNet_head(self): 
        input_shape=self.img_input

        conv1 = Conv2D(96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',strides=(1, 1), padding='same', name='conv1',data_format="channels_last")(input_shape)
        maxpool1 = MaxPooling2D(pool_size=(3,3), strides=(2, 2),padding='same',  name='maxpool1',data_format="channels_last")(conv1)
        f1=maxpool1

        fire2_squeeze = Conv2D(16, (1, 1), activation='relu', kernel_initializer='glorot_uniform', padding='same', name='fire2_squeeze',data_format="channels_last")(maxpool1)
        fire2_expand1 = Conv2D(64, (1, 1), activation='relu', kernel_initializer='glorot_uniform', padding='same', name='fire2_expand1',data_format="channels_last")(fire2_squeeze)
        fire2_expand2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same', name='fire2_expand2', data_format="channels_last")(fire2_squeeze)
        merge2 = Concatenate(axis=-1)([fire2_expand1, fire2_expand2])

        fire3_squeeze = Conv2D(16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire3_squeeze',data_format="channels_last")(merge2)
        fire3_expand1 = Conv2D(64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire3_expand1',data_format="channels_last")(fire3_squeeze)
        fire3_expand2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire3_expand2',data_format="channels_last")(fire3_squeeze)
        merge3 = Concatenate(axis=-1)([fire3_expand1, fire3_expand2])
        maxpool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool2', data_format="channels_last")(merge3)
        f2=maxpool2

        fire4_squeeze = Conv2D(32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire4_squeeze',data_format="channels_last")(maxpool2)
        fire4_expand1 = Conv2D(128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire4_expand1',data_format="channels_last")(fire4_squeeze)
        fire4_expand2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire4_expand2',data_format="channels_last")(fire4_squeeze)
        merge4 = Concatenate(axis=-1)([fire4_expand1, fire4_expand2])                           

        fire5_squeeze = Conv2D(32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire5_squeeze',data_format="channels_last")(merge4)
        fire5_expand1 = Conv2D(128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire5_expand1',data_format="channels_last")(fire5_squeeze)
        fire5_expand2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire5_expand2',data_format="channels_last")(fire5_squeeze)
        merge5 = Concatenate(axis=-1)([fire5_expand1, fire5_expand2])   
        maxpool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool3', data_format="channels_last")(merge5)
        f3=maxpool3

        fire6_squeeze = Conv2D(48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire6_squeeze',data_format="channels_last")(maxpool3)
        fire6_expand1 = Conv2D(192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire6_expand1',data_format="channels_last")(fire6_squeeze)
        fire6_expand2 = Conv2D(192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire6_expand2',data_format="channels_last")(fire6_squeeze)
        merge6 = Concatenate(axis=-1)([fire6_expand1, fire6_expand2])

        fire7_squeeze = Conv2D(48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire7_squeeze',data_format="channels_last")(merge6)
        fire7_expand1 = Conv2D(192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire7_expand1',data_format="channels_last")(fire7_squeeze)
        fire7_expand2 = Conv2D(192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire7_expand2',data_format="channels_last")(fire7_squeeze)
        merge7 = Concatenate(axis=-1)([fire7_expand1, fire7_expand2])
        maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool4', data_format="channels_last")(merge7)
        f4=maxpool4    

        fire8_squeeze = Conv2D(64, (1, 1), activation='relu', kernel_initializer='glorot_uniform', padding='same', name='fire8_squeeze',data_format="channels_last")(maxpool4)
        fire8_expand1 = Conv2D(256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire8_expand1',data_format="channels_last")(fire8_squeeze)
        fire8_expand2 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire8_expand2',data_format="channels_last")(fire8_squeeze)
        merge8 = Concatenate(axis=-1)([fire8_expand1, fire8_expand2])

        maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool8', data_format="channels_last")(merge8)
        fire9_squeeze = Conv2D(64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire9_squeeze',data_format="channels_last")(maxpool8)
        fire9_expand1 = Conv2D(256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire9_expand1',data_format="channels_last")(fire9_squeeze)
        fire9_expand2 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',padding='same', name='fire9_expand2',data_format="channels_last")(fire9_squeeze)
        merge9 = Concatenate(axis=-1)([fire9_expand1, fire9_expand2])
    #     fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
        f5=merge9     

#         conv10 = Conv2D(nb_classes, (1, 1), activation='relu', kernel_initializer='glorot_uniform',padding='valid', name='conv10',data_format="channels_last")(fire9_dropout)
#         global_avgpool10 = GlobalAveragePooling2D(data_format='channels_last')(conv10)
#         result = Activation("sigmoid", name='softmax')(global_avgpool10)
#         squeesenetmodel=Model(inputs=input_img, outputs=result)

        return input_shape,[f1, f2, f3, f4, f5]    

   ## XCEPTION_head2(deeplabv3+)
    def Xception_head2(self):
        img_input=self.img_input
        ## enter flow
        x = Conv2D(32, (3, 3), strides=(2, 2),padding='same', use_bias=False, name='block1_conv1')(img_input)
        x = BatchNormalization(name='block1_conv1_bn')(x)
        x = Activation('relu', name='block1_conv1_act')(x)
        x = Conv2D(64, (3, 3), padding='same',use_bias=False, name='block1_conv2')(x)
        x = BatchNormalization(name='block1_conv2_bn')(x)
        x = Activation('relu', name='block1_conv2_act')(x)
        f1=x#128
        residual = Conv2D(128, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
        x = BatchNormalization(name='block2_sepconv1_bn')(x)
        x = Activation('relu', name='block2_sepconv2_act')(x)
        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
        x = BatchNormalization(name='block2_sepconv2_bn')(x)
        x = SeparableConv2D(128, (3, 3), padding='same', strides=(2, 2), name='block2_pool')(x)
        x = add([x, residual])
        
        f2=x#64
        residual = Conv2D(256, (1, 1), strides=(1, 1),padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu', name='block3_sepconv1_act')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
        x = BatchNormalization(name='block3_sepconv1_bn')(x)
        x = Activation('relu', name='block3_sepconv2_act')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
        x = BatchNormalization(name='block3_sepconv2_bn')(x)

        x = SeparableConv2D(256, (3, 3), padding='same',strides=(1, 1), name='block3_pool')(x)
        x = add([x, residual])
        f3=x#64
        residual = Conv2D(728, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu', name='block4_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
        x = BatchNormalization(name='block4_sepconv1_bn')(x)
        x = Activation('relu', name='block4_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
        x = BatchNormalization(name='block4_sepconv2_bn')(x)

        x = SeparableConv2D(728, (3, 3), padding='same',strides=(2, 2), name='block4_pool')(x)
        x = add([x, residual])
        ## middle flow
        for i in range(16):
            residual = x
            prefix = 'block' + str(i + 5)

            x = Activation('relu', name=prefix + '_sepconv1_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
            x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
            x = Activation('relu', name=prefix + '_sepconv2_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
            x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
            x = Activation('relu', name=prefix + '_sepconv3_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
            x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

            x = add([x, residual])

        ## exit flow
        residual = Conv2D(1024, (1, 1), strides=(1,1),padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu',name='block21_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same',dilation_rate=(2, 2), use_bias=False, name='block21_sepconv1')(x)
        x = BatchNormalization(name='block21_sepconv1_bn')(x)
        x = Activation('relu', name='block21_sepconv2_act')(x)
        x = SeparableConv2D(1024, (3, 3), padding='same',dilation_rate=(2, 2),use_bias=False, name='block21_sepconv2')(x)
        x = BatchNormalization(name='block21_sepconv2_bn')(x)
        f4=x #32
        # the last stage stride=1,which is different with the original xception.
        x = SeparableConv2D(1024, (3, 3), strides=(1,1), dilation_rate=(2, 2),padding='same', name='block21_pool')(x)
        x = add([x, residual])

        x = SeparableConv2D(1536, (3, 3), padding='same',dilation_rate=(4, 4), use_bias=False, name='block22_sepconv1')(x)
        x = BatchNormalization(name='block22_sepconv1_bn')(x)
        x = Activation('relu', name='block22_sepconv1_act')(x)

        x = SeparableConv2D(1536, (3, 3), padding='same',dilation_rate=(4, 4), use_bias=False, name='block23_sepconv1')(x)
        x = BatchNormalization(name='block23_sepconv1_bn')(x)
        x = Activation('relu', name='block23_sepconv1_act')(x)

        x = SeparableConv2D(2048, (3, 3), padding='same', dilation_rate=(4, 4),use_bias=False, name='block24_sepconv2')(x)
        x = BatchNormalization(name='block24_sepconv2_bn')(x)
        x = Activation('relu', name='block24_sepconv2_act')(x)
        f5=x #32
#         fgap = GlobalAveragePooling2D(name='avg_pool')(x)
#         x = Dense(1, activation='sigmoid', name='predictions')(x)

#         model = Model(self.img_input, x, name='xception')
        return img_input,[f1, f2, f3, f4, f5] 
   ## FCN8 
    def FCN8(self):
        [f1, f2, f3, f4, f5,f6,f7,f8]=self.VGG16_head()

        # Conv to be applied on Pool4
        skip_con1 = Conv2D(self.nClasses, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",name="score_pool4")(f4)
        Summed = add(inputs=[skip_con1, f8])
        # size conver from 16 to 32
        x = Conv2DTranspose(self.nClasses, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None, name="score4")(Summed)

        ###
        skip_con2 = Conv2D(self.nClasses, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",name="score_pool3")(f3)
        Summed2 = add(inputs=[skip_con2, x])

        ##### size conver from 32 to 256
        Up = Conv2DTranspose(self.nClasses, kernel_size=(8, 8), strides=(8, 8),padding="valid", activation=None, name="upsample")(Summed2)
#         Up = Activation("sigmoid")(Up)
      
        Up = Conv2D(self.nClasses, kernel_size=(1, 1),padding = 'same', activation = 'softmax')(Up)

        self.fcn8model = Model(inputs=self.img_input, outputs=Up)
        self.fcn8model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)
        return self.fcn8model
    ## UNET
    def UNET_VGG(self):
    #     Patch_size = 224
        conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(self.img_input)
        conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

        up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
        merge6 = Concatenate(axis = -1)([conv4, up6])
        conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = Concatenate(axis = -1)([conv3,up7])
        conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = Concatenate(axis = -1)([conv2,up8])
        conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = Concatenate(axis = -1)([conv1,up9])
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

        conv10 = Conv2D(self.nClasses, 1, activation = 'softmax',padding = 'same')(conv9)

        self.unetmodel = Model(inputs=self.img_input, outputs=conv10)
        self.unetmodel.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)
        return self.unetmodel
   ## UNET_MINI
    def UNET_MINI(self):

        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(self.img_input)
        conv1 = Dropout(0.2)(conv1)
        conv1 = Conv2D(32, (3, 3),activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Dropout(0.2)(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Dropout(0.2)(conv3)
        conv3 = Conv2D(128, (3, 3),activation='relu', padding='same')(conv3)

        up1 = Concatenate(axis=-1)([UpSampling2D((2, 2), )(conv3), conv2])
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
        conv4 = Dropout(0.2)(conv4)
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

        up2 = Concatenate(axis=-1)([UpSampling2D((2, 2))( conv4), conv1])
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
        conv5 = Dropout(0.2)(conv5)
        conv5 = Conv2D(32, (3, 3),activation='relu', padding='same')(conv5)

        result = Conv2D(self.nClasses, 1, activation = 'softmax',padding = 'same')(conv5)

        self.unetminimodel = Model(inputs=self.img_input, outputs=result)
        self.unetminimodel.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)
        return self.unetminimodel
 
   ## SQUEESE_UNET
    def SQUEESE_UNET(self):

        SQUEESE_input, levels = self.SqueezeNet_head()
        [f1, f2, f3, f4, f5] = levels  
        o = f5        
        o = (UpSampling2D((2, 2)))(f5)
        o = (Concatenate(axis = -1)([o, f4]))

#         o = (ZeroPadding2D((1, 1)))(o)
        o = (Conv2D(512, (3, 3), padding='same'))(o)
        o = (BatchNormalization())(o)

        o = (UpSampling2D((2, 2)))(o)
        o = (Concatenate(axis = -1)([o, f3]))
#         o = (ZeroPadding2D((1, 1)))(o)
        o = (Conv2D(256, (3, 3), padding='same'))(o)
        o = (BatchNormalization())(o)

        o = (UpSampling2D((2, 2)))(o)
        o = (Concatenate(axis = -1)([o, f2]))
#         o = (ZeroPadding2D((1, 1)))(o)
        o = (Conv2D(128, (3, 3), padding='same'))(o)
        o = (BatchNormalization())(o)

        o = (UpSampling2D((2, 2)))(o)
        o = (Concatenate(axis = -1)([o, f1]))
#         o = (ZeroPadding2D((1, 1)))(o)
        o = (Conv2D(64, (3, 3), padding='same'))(o)
        o = (BatchNormalization())(o)

        o = (UpSampling2D((2, 2)))(o)
        o = (Concatenate(axis = -1)([o, SQUEESE_input]))
#         o = (ZeroPadding2D((1, 1)))(o)
        o = (Conv2D(64, (3, 3), padding='same'))(o)
        o = (BatchNormalization())(o)
        o = Dropout(0.5)(o)
        
        result = Conv2D(self.nClasses, 1, activation = 'softmax',padding = 'same')(o)

        self.squeeseunetmodel = Model(inputs=self.img_input, outputs=result)
        self.squeeseunetmodel.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)
        return self.squeeseunetmodel
   ## CBRRNET
    def CBRR_BLOCK(self,x,f_kernel_size,filters,dilation,pad=None):

        filters_1, filters_2, filters_3 = filters

        # stage 1
        x = Conv2D(filters=filters_1, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(momentum=0.95, axis=-1)(x)
        x = Activation(activation='relu')(x)
        x_shortcut=x

        # stage 2
        #x = ZeroPadding2D(padding=pad)(x)
        x = Conv2D(filters=filters_2, kernel_size=f_kernel_size, strides=(1, 1),padding='same',
                   dilation_rate=dilation, kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(momentum=0.95, axis=-1)(x)
        x = Activation(activation='relu')(x)

        # stage 3
        x = Conv2D(filters=filters_2, kernel_size=(1,1), strides=(1, 1),padding='same',
                    kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(momentum=0.95, axis=-1)(x)
        x = Activation(activation='relu')(x)

        # stage 4
        x=add([x,x_shortcut])
        #x=Activation(activation='relu')(x)

        return x
    def Encoder(self,inputs):
        # inputs=Input(shape(22,224,3))

        #stage 1 第一次下采样
        x_stage_1=self.CBRR_BLOCK(inputs,f_kernel_size=(3,3),filters=[64,64,64],dilation=1)
        x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x_stage_1)

        #stage 2   第二次下采样
        x_stage_2=self.CBRR_BLOCK(x,f_kernel_size=(3,3),filters=[128,128,128],dilation=1)
        x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x_stage_2)

        #stage 3   第三次下采样
        x_stage_3=self.CBRR_BLOCK(x,f_kernel_size=(3,3),filters=[256,256,256],dilation=1)
        x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x_stage_3)

        #stage 4,5,6
        x_stage_4=self.CBRR_BLOCK(x,f_kernel_size=(3,3),filters=[512,512,512],dilation=1)
        x_stage_5=self.CBRR_BLOCK(x_stage_4,f_kernel_size=(3,3),filters=[512,512,512],dilation=2)
        x_stage_6=self.CBRR_BLOCK(x_stage_5,f_kernel_size=(3,3),filters=[512,512,512],dilation=4)

        return x_stage_1,x_stage_2,x_stage_3,x_stage_4,x_stage_5,x_stage_6
    def CBRRNET(self):
        inputs=self.img_input
        shape=np.array([self.input_height,self.input_width]).astype(int)
        
        x_stage_1,x_stage_2,x_stage_3,x_stage_4,x_stage_5,x_stage_6=self.Encoder(inputs)

        x_c6=self.CBRR_BLOCK(x_stage_6,f_kernel_size=(3,3),filters=[512,512,512],dilation=4)  #第一个输出分支
        #skip connection
        x_c6=Concatenate(axis=-1,name="concat_6")([x_c6,x_stage_6])

        x_c5=self.CBRR_BLOCK(x_c6,f_kernel_size=(3,3),filters=[512,512,512],dilation=2)  #第二个输出分支
        x_c5=Concatenate(axis=-1,name="concat_5")([x_c5,x_stage_5])

        x_c4=self.CBRR_BLOCK(x_c5,f_kernel_size=(3,3),filters=[512,512,512],dilation=1)  #第三个输出分支
        x_c4=Concatenate(axis=-1,name="concat_4")([x_c4,x_stage_4])

        x_c3=Lambda(self.interpolation,arguments={'shape':(shape/4).astype(int)})(x_c4)
        #x_c3= UpSampling2D(size=(2, 2))(x_c4)
        x_c3=self.CBRR_BLOCK(x_c3,f_kernel_size=(3,3),filters=[256,256,256],dilation=1)  #第四个输出分支
        x_c3=Concatenate(axis=-1,name="concat_3")([x_c3,x_stage_3])

        x_c2=Lambda(self.interpolation,arguments={'shape':(shape/2).astype(int)})(x_c3)
        #x_c2= UpSampling2D(size=(2, 2))(x_c3)
        x_c2=self.CBRR_BLOCK(x_c2,f_kernel_size=(3,3),filters=[128,128,128],dilation=1)  #第五个输出分支
        x_c2=Concatenate(axis=-1,name="concat_2")([x_c2,x_stage_2])

        x_c1=Lambda(self.interpolation,arguments={'shape':shape})(x_c2)
        #x_c1= UpSampling2D(size=(2, 2))(x_c2)
        x_c1=self.CBRR_BLOCK(x_c1,f_kernel_size=(3,3),filters=[64,64,64],dilation=1)  #第五个输出分支
        x_c1=Concatenate(axis=-1,name="concat_1")([x_c1,x_stage_1])

        """output 6 path"""
        output_6=Conv2D(filters=64,kernel_size=3,strides=1,padding="same",name="output_6")(x_c6)
        output_6= BatchNormalization(momentum=0.95, axis=-1)(output_6)
        output_6 = Activation(activation='relu')(output_6)
        output_6=Lambda(self.interpolation,arguments={'shape':shape})(output_6)

        output_5=Conv2D(filters=64,kernel_size=3,strides=1,padding="same",name="output_5")(x_c5)
        output_5= BatchNormalization(momentum=0.95, axis=-1)(output_5)
        output_5 = Activation(activation='relu')(output_5)
        output_5=Lambda(self.interpolation,arguments={'shape':shape})(output_5)

        output_4=Conv2D(filters=64,kernel_size=3,strides=1,padding="same",name="output_4")(x_c4)
        output_4= BatchNormalization(momentum=0.95, axis=-1)(output_4)
        output_4 = Activation(activation='relu')(output_4)
        output_4=Lambda(self.interpolation,arguments={'shape':shape})(output_4)

        output_3=Conv2D(filters=64,kernel_size=3,strides=1,padding="same",name="output_3")(x_c3)
        output_3= BatchNormalization(momentum=0.95, axis=-1)(output_3)
        output_3 = Activation(activation='relu')(output_3)
        output_3=Lambda(self.interpolation,arguments={'shape':shape})(output_3)

        output_2=Conv2D(filters=64,kernel_size=3,strides=1,padding="same",name="output_2")(x_c2)
        output_2= BatchNormalization(momentum=0.95, axis=-1)(output_2)
        output_2 = Activation(activation='relu')(output_2)
        output_2=Lambda(self.interpolation,arguments={'shape':shape})(output_2)

        output_1=Conv2D(filters=64,kernel_size=3,strides=1,padding="same",name="output_1")(x_c1)
        output_1= BatchNormalization(momentum=0.95, axis=-1)(output_1)
        output_1 = Activation(activation='relu')(output_1)

        outputs=Concatenate(axis=-1,name="final_concat")([output_6,output_5,output_4,output_3,output_2,output_1])

        output_s=Conv2D(filters=64,kernel_size=3,strides=1,padding="same",name="outputs_pre")(outputs)
        output_s= BatchNormalization(momentum=0.95, axis=-1)(output_s)
        output_s = Activation(activation='relu')(output_s)

        outputs=Conv2D(filters=self.nClasses,kernel_size=1,strides=1,padding="same",name="final_outputs")(outputs)
        outputs = Activation(activation='softmax')(outputs)


        self.cbrrmodel=Model(inputs=inputs,outputs=outputs)
        self.cbrrmodel.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)
        return self.cbrrmodel
## DEEPLAB V3 PLUS

    def interpolation(self,x, shape,method=0):
        # """0：双线性差值。1：最近邻居法。2：双三次插值法。3：面积插值法"""
        import tensorflow as tf
        # The height and breadth to which the pooled feature maps are to be interpolated
        h_to, w_to = shape
        # Bilinear Interpolation (Default method of this tf function is method=ResizeMethod.BILINEAR)
        resized = tf.image.resize(x, [h_to, w_to])
        return resized
    def SepConv_BN(self,x, filters, prefix, stride=1, kernel_size=3, rate=1, epsilon=1e-5):

        x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),padding='same', use_bias=False, name=prefix + '_depthwise')(x)
        x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (1, 1), padding='same',use_bias=False, name=prefix + '_pointwise')(x)
        x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
        x = Activation('relu')(x)

        return x
    def DEEPLABV3plus(self):
        inputs=self.img_input
        XCEPTION_input2, levels = self.Xception_head2()
        [f1, f2, f3, f4, f5] = levels
#         print(K.int_shape(f3),K.int_shape(f4),K.int_shape(f5))
        #F3 64,F4=F5=32
        b4 = GlobalAveragePooling2D()(f5)

        # from (b_size, channels)->(b_size, 1, 1, channels)
        b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
        b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
        b4 = Conv2D(256, (1, 1), padding='same',use_bias=False, name='image_pooling')(b4)
        b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
        b4 = Activation('relu')(b4)
        # upsample. have to use compat because of the option align_corners
        size_before = K.int_shape(f5)
        b4 = Lambda(self.interpolation, arguments={'shape': (size_before[1], size_before[2])})(b4)
#         print(K.int_shape(b4))

    # simple 1x1
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(f5)
        b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        b0 = Activation('relu', name='aspp0_activation')(b0)    
        # Dilated conv block
        atrous_rates = (6, 12, 18)

        # rate = 6 (12)
        b1 = self.SepConv_BN(f5, 256, 'aspp1',rate=atrous_rates[0])
        # rate = 12 (24)
        b2 = self.SepConv_BN(f5, 256, 'aspp2',rate=atrous_rates[1])
        # rate = 18 (36)
        b3 = self.SepConv_BN(f5, 256, 'aspp3',rate=atrous_rates[2])

        # concatenate ASPP branches & project
        x = Concatenate()([b4, b0, b1, b2, b3])
        x = Conv2D(256, (1, 1), padding='same',use_bias=False, name='concat_projection')(x)
        x = BatchNormalization(name='concat_projection_BN')(x)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)
        # X=32
        # DeepLab v.3+ decoder
#         size_before2 = tf.keras.backend.int_shape(f4)
#         x = Lambda(self.interpolation, arguments={'shape': (size_before2[1], size_before2[2])})(x)
        x = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear', name='4x_Upsampling1')(x)

        dec_skip1 = Conv2D(48, (1, 1), padding='same',use_bias=False, name='feature_projection0')(f3)
        dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation(tf.nn.relu)(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        # X=64
        x = self.SepConv_BN(x, 256, 'decoder_conv0')
        x = self.SepConv_BN(x, 256, 'decoder_conv1')      
        x = UpSampling2D(size=(4,4), data_format='channels_last', interpolation='bilinear', name='4x_Upsampling2')(x)
        # X=256
#         x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
#         size_before3 = tf.keras.backend.int_shape(inputs)
#         x = Lambda(self.interpolation, arguments={'shape': (size_before3[1], size_before3[2])})(x)

        outputs=Conv2D(filters=self.nClasses,kernel_size=1,strides=1,padding="same",name="final_outputs")(x)
        outputs = Activation(activation='softmax')(outputs)

        self.deeplabv3model=Model(inputs=inputs,outputs=outputs)
        self.deeplabv3model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)
        return self.deeplabv3model