#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:09:00 2017

@author: liuhuihui
"""
from keras import regularizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.utils import np_utils
from keras.optimizers import SGD,Adam,Adadelta
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping,LearningRateScheduler
import matplotlib.pyplot as plt
import h5py
import random
from keras.layers import Merge
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

"""
#AlexNet
def build_social_net(input_shape,num_classes):
    model=Sequential()
    
    model.add(Conv2D(48,(11,11),strides=(4,4), data_format = 'channels_last', input_shape = input_shape,init='uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(BatchNormalization(axis=1,epsilon=0.001))

    model.add(Conv2D(128,(5,5),strides=(1,1),padding='valid',init='uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization(axis=1,epsilon=0.001))

    model.add(Conv2D(192,(3,3),strides=(1,1),init='uniform'))
    model.add(Activation('relu'))
    
    model.add(Conv2D(192,(3,3),strides=(1,1),init='uniform'))
    model.add(Activation('relu'))
    
    model.add(Conv2D(128,(3,3),strides=(1,1),init='uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(Flatten())
    
    model.add(Dense(2048,init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(2048,init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    return model
"""

#Lu
def build_net(input_shape, num_classes):
    model=Sequential()
    
    model.add(Conv2D(64,(11,11),strides=(2,2), data_format = 'channels_last', input_shape = input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization(axis=-1,epsilon=0.001))

    model.add(Conv2D(64,(5,5),padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization(axis=-1,epsilon=0.001))

    model.add(Conv2D(64,(3,3),strides=(1,1)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(64,(3,3),strides=(1,1)))
    model.add(Activation('relu'))

    model.add(Flatten())
    
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    return model

def generate_sequences(n_batches, images, labels, mean, idxs):
    # generate batches of samples
    
    while True:
        for bid in xrange(0, n_batches):
            if bid == n_batches - 1:
                batch_idxs = idxs[bid * batch_size:]
            else:
                batch_idxs = idxs[bid * batch_size: (bid + 1) * batch_size]
                                  
            batch_length=len(batch_idxs)
            X = np.zeros((batch_length,224,224,3),np.float32)            
                     
            y = labels[0][batch_idxs]
            Y = np_utils.to_categorical(y, 2)
          
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
            
            yield [X,X],Y


if __name__ == '__main__':

    input_shape = (224, 224, 3)
   
    model1 = build_social_net(input_shape,1000)
    model1.load_weights('/home/liuhuihui/ME/model/model-social-AlexNet.h5',by_name=True)
    for layer in model1.layers:
        layer.trainable=False
    
    model2 = build_social_net(input_shape, 2)
    
    model=Sequential()
    model.add(Merge([model1,model2],mode='concat'))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=0.001, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(optimizer=sgd,loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    batch_size = 64
    nb_epoch = 150
    validation_ratio = 0.1
    
    
    #model.load_weights('/home/liuhuihui/ME/data/model_dcnn_social.h5',by_name=True)

    # training
    path_train = '/home/liuhuihui/DMA_Net/Warp/train_data.hdf5'
    with h5py.File(path_train, 'r') as train_file:
        images = train_file['images']
        labels = train_file['labels'][...]
        labels=labels[np.newaxis,...]
        print labels.shape
        
        mean = train_file['mean'][...]
    
        idxs = range(len(images))
        train_idxs = idxs[: int(len(images) * (1 - validation_ratio))]
        validation_idxs = idxs[int(len(images) * (1 - validation_ratio)) :]

        # training sample generator
        n_train_batches = len(train_idxs) // batch_size
        n_remainder = len(train_idxs) % batch_size
        if n_remainder:
            n_train_batches = n_train_batches + 1
        train_generator = generate_sequences(n_train_batches, images, labels,mean, train_idxs)

        # validation sample generator
        n_validation_batches = len(validation_idxs) // batch_size
        n_remainder = len(validation_idxs) % batch_size
        if n_remainder:
            n_validation_batches = n_validation_batches + 1
        validation_generator = generate_sequences(n_validation_batches, images, labels, mean,validation_idxs)

        #LearningRateS=LearningRateScheduler(step_decay)
        
        ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10,
                 verbose=1, mode='min', epsilon=1e-4, cooldown=0, min_lr=0)
        
        Checkpoint = ModelCheckpoint(filepath='/home/liuhuihui/ME/data/model_dcnn_social2.h5', monitor='val_loss', verbose=1,
                 save_best_only=True, save_weights_only=False,mode='min', period=1)
        
        earlyStop = EarlyStopping(monitor='val_loss',patience=15,verbose=1,mode='min')
        
        history = model.fit_generator(generator = train_generator,
                            validation_data = validation_generator,
                            steps_per_epoch = n_train_batches,
                            validation_steps = n_validation_batches,
                            epochs = nb_epoch,
                            callbacks=[ReduceLR,Checkpoint,earlyStop])
        
        
    
    path_test='/home/liuhuihui/DMA_Net/Warp/test_data.hdf5'
    with h5py.File(path_test, 'r') as test_file:
        images = test_file['images']
        labels = test_file['labels'][...]
        labels=labels[np.newaxis,...]
        #labels = np_utils.to_categorical(labels, 2) 

        idxs = range(len(images))
        test_idxs = idxs[: int(len(images))]
                         
        # testing sample generator
        n_test_batches = len(test_idxs) // batch_size
        n_remainder = len(test_idxs) % batch_size
        if n_remainder:
            n_test_batches = n_test_batches + 1
        test_generator = generate_sequences(n_test_batches, images, labels, mean, test_idxs)        
        predicted = model.predict_generator(generator=test_generator,steps=n_test_batches,workers=1,verbose=1)      
        print predicted      
        
        score=model.evaluate_generator(generator=test_generator,steps=n_test_batches,workers=1)
        print score
     