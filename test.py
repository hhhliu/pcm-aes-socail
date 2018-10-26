#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:33:54 2017

@author: liuhuihui
"""

from keras import regularizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.utils import np_utils
from keras.optimizers import SGD,Adam,Adadelta,RMSprop
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping,LearningRateScheduler
import matplotlib.pyplot as plt
from keras.layers import Input, Flatten, Dense
from keras.applications.resnet50 import ResNet50
#from resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.models import Model
import h5py
import random
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb
import keras.backend as K
from keras.utils import plot_model
from keras.layers import Merge
from keras.preprocessing.image import ImageDataGenerator
import losse_Hui
import weightLoss
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

def generate_sequences(n_batches, images, labels_tag, labels_user, labels_group, mean, idxs):
    # generate batches of samples
    
    while True:
        for bid in xrange(0, n_batches):
            if bid == n_batches - 1:
                batch_idxs = idxs[bid * batch_size:]
            else:
                batch_idxs = idxs[bid * batch_size: (bid + 1) * batch_size]
                                  
            batch_length=len(batch_idxs)
            X = np.zeros((batch_length,224,224,3),np.float32)                              
            Y_tag = labels_tag[batch_idxs,:]
            Y_user = labels_user[batch_idxs,:]
            Y_group = labels_group[batch_idxs,:]
            
            count=0
            # for every image of a batch
            for i in batch_idxs:
                xx=images[i,...].astype(np.float32)
                xx-=mean
                offset_x=random.randint(0,31)
                offset_y=random.randint(0,31)
                xx=xx[offset_x:offset_x+224,offset_y:offset_y+224,:]                
                X[count,...]=xx
                count+=1                
            datagen = ImageDataGenerator(#featurewise_center=True,
#                                         featurewise_std_normalization=True,
#                                         rotation_range=30,
#                                         width_shift_range=0.08,
#                                         height_shift_range=0.08,
                                         shear_range=0.2,
#                                         zoom_range=0.2,
                                         horizontal_flip=True,
                                         vertical_flip=True,                                         
                                         )            
            datagen.fit(X)
            i=1
            for batch in datagen.flow(X,batch_size=32,shuffle=False):
                yield batch,[Y_tag,Y_user,Y_group]
                i += 1
                if i > 20:
                    break

if __name__ == '__main__':
    
    ''' 
    input_shape = (224, 224, 3)
    model_vgg16=VGG16(weights=None,include_top=False)
    model_vgg16.load_weights('/home/liuhuihui/ME/model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    for layer in model_vgg16.layers:
        layer.trainable = False
    input = Input(shape=input_shape,name = 'image_input')
    output_vgg16_conv = model_vgg16(input)
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    '''   
    
    input_shape = (224, 224, 3)
    input_tensor = Input(shape=input_shape)
    resnet50 = ResNet50(include_top=False, weights=None,input_tensor=input_tensor)
    resnet50.load_weights('/home/liuhuihui/ME/model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',by_name=True)
#    for layer in resnet50.layers:
#        layer.trainable=False         
    output_resnet_conv_tag = resnet50(input_tensor)
    x = Flatten(name='flatten')(output_resnet_conv_tag)
    
    model_tag=Dense(4096, activation='relu',name='tag1')(x)
    model_tag = Dropout(0.5)(model_tag)
    model_tag=Dense(2048, activation='relu',name='tag2')(model_tag)
    model_tag = Dropout(0.5)(model_tag)
    model_tag=Dense(399, activation='softmax',name='tag3')(model_tag)
    model_tag = Dropout(0.5)(model_tag)
    
    model_user=Dense(4096, activation='relu',name='user1')(x)
    model_user = Dropout(0.5)(model_user)
    model_user=Dense(2048, activation='relu',name='user2')(model_user)
    model_user = Dropout(0.5)(model_user)
    model_user=Dense(240, activation='softmax',name='user3')(model_user)
    model_user = Dropout(0.5)(model_user)
    
    model_group=Dense(4096, activation='relu',name='group1')(x) 
    model_group = Dropout(0.5)(model_group)
    model_group=Dense(2048, activation='relu',name='group2')(model_group) 
    model_group = Dropout(0.5)(model_group)
    model_group=Dense(327, activation='softmax',name='group3')(model_group) 
    model_group = Dropout(0.5)(model_group)
    
    model=Model(input=input_tensor,output=[model_tag,model_user,model_group])
    model.summary()
    
    def my_acc(y_true,y_pred):
        correct_prediction = tf.equal(tf.round(y_pred), y_true)        
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return evaluation_step
    
    sgd = SGD(lr=0.0001, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(optimizer=sgd,loss='categorical_crossentropy')   
    #model.compile(optimizer=sgd,loss=[weightLoss.tag_loss(0.2),weightLoss.user_loss(0.6),weightLoss.group_loss(0.2)])   
    batch_size = 32
    nb_epoch = 300
    validation_ratio = 0.1
    
    path_train = '/home/liuhuihui/ME/newData/train_flickr.hdf5'
    with h5py.File(path_train, 'r') as train_file:
        images = train_file['images']
        labels_tag = train_file['labels_tag_01'][...]
        labels_user= train_file['labels_user_01'][...]
        labels_group = train_file['labels_group_01'][...]
        mean = train_file['mean'][...]
    
        idxs = range(len(images))
        train_idxs = idxs[: int(len(images) * (1 - validation_ratio))]
        validation_idxs = idxs[int(len(images) * (1 - validation_ratio)) :]

        n_train_batches = len(train_idxs) // batch_size
        n_remainder = len(train_idxs) % batch_size
        if n_remainder:
            n_train_batches = n_train_batches + 1
        train_generator = generate_sequences(n_train_batches, images, labels_tag, labels_user, labels_group, mean, train_idxs)

        n_validation_batches = len(validation_idxs) // batch_size
        n_remainder = len(validation_idxs) % batch_size
        if n_remainder:
            n_validation_batches = n_validation_batches + 1
        validation_generator = generate_sequences(n_validation_batches, images, labels_tag, labels_user, labels_group, mean,validation_idxs)
        
        ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8,
                 verbose=1, mode='min', epsilon=1e-4, cooldown=0, min_lr=0)
        
        Checkpoint = ModelCheckpoint(filepath='/home/liuhuihui/ME/newData/newmodel/model_final41-cross-01-resnet50-DA-noweight.h5', monitor='val_loss', verbose=1,
                 save_best_only=True, save_weights_only=False,mode='min', period=1)
        
        earlyStop = EarlyStopping(monitor='val_loss',patience=15,verbose=1,mode='min')
        
        history = model.fit_generator(generator = train_generator,
                            validation_data = validation_generator,
                            steps_per_epoch = n_train_batches,
                            validation_steps = n_validation_batches,
                            epochs = nb_epoch,
                            callbacks=[ReduceLR,Checkpoint,earlyStop])        
        
    
    path_test='/home/liuhuihui/ME/newData/test_flickr.hdf5'
    with h5py.File(path_test, 'r') as test_file:
        images = test_file['images']
        labels_tag = test_file['labels_tag_01'][...]
        labels_user= test_file['labels_user_01'][...]
        labels_group = test_file['labels_group_01'][...]
        mean = test_file['mean'][...]
        
        idxs = range(len(images))
        test_idxs = idxs[: int(len(images))]

        n_test_batches = len(test_idxs) // batch_size
        n_remainder = len(test_idxs) % batch_size
        if n_remainder:
            n_test_batches = n_test_batches + 1
        test_generator = generate_sequences(n_test_batches, images, labels_tag, labels_user, labels_group, mean, test_idxs)        
        predicted = model.predict_generator(generator=test_generator,steps=n_test_batches,workers=1,verbose=1)      
        print predicted  
        print type(predicted)

        score=model.evaluate_generator(generator=test_generator,steps=n_test_batches,workers=1)
        print score
