# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:01:59 2018

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
from keras.layers import Input, Flatten, Dense
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.models import Model
import h5py
import random
from sklearn.metrics import roc_curve,roc_auc_score
from keras.layers.merge import concatenate
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb
import keras.backend as K
from keras.utils import plot_model
from keras.layers import Merge
import scipy.io as sio
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

def generate_sequences(n_batches, images, labels, mean, idxs):
    
    while True:
        for bid in xrange(0, n_batches):
            if bid == n_batches - 1:
                batch_idxs = idxs[bid * batch_size:]
            else:
                batch_idxs = idxs[bid * batch_size: (bid + 1) * batch_size]
                                  
            batch_length=len(batch_idxs)
            X = np.zeros((batch_length,224,224,3),np.float32)            
                     
            y = labels[batch_idxs]
            Y = np_utils.to_categorical(y, 2)
          
            count=0
            for i in batch_idxs:
                xx=images[i,...].astype(np.float32)
                xx-=mean
                offset_x=random.randint(0,31)
                offset_y=random.randint(0,31)
                xx=xx[offset_x:offset_x+224,offset_y:offset_y+224,:]
                
                X[count,...]=xx
                count+=1    
            
            yield X,Y

if __name__ == '__main__':
    
    input_shape = (224, 224, 3)
    input_tensor = Input(shape=input_shape)
    resnet50 = ResNet50(include_top=False, weights=None,input_tensor=input_tensor)
    resnet50.load_weights('/home/liuhuihui/ME/data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',by_name=True)       
    x = resnet50(input_tensor)
    x = Flatten(name='flatten')(x)
    x = Dense(2, activation='softmax')(x)
    model=Model(outputs=x,inputs=input_tensor)
    model.summary()
    sgd = SGD(lr=0.0001, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(optimizer=sgd,loss='binary_crossentropy', metrics=['accuracy'])
    batch_size = 32
    nb_epoch = 300
    validation_ratio = 0.1

    #path_train = '/home/liuhuihui/ME/newData/ava_delta0/train_ava.hdf5'
    path_train='/home/liuhuihui/ME/data/ava/train_data.hdf5'
    with h5py.File(path_train, 'r') as train_file:
        images = train_file['images']
        labels = train_file['labels'][...]
        mean = train_file['mean'][...]
    
        idxs = range(len(images))
        train_idxs = idxs[: int(len(images) * (1 - validation_ratio))]
        validation_idxs = idxs[int(len(images) * (1 - validation_ratio)) :]

        # training sample generator
        n_train_batches = len(train_idxs) // batch_size
        n_remainder = len(train_idxs) % batch_size
        if n_remainder:
            n_train_batches = n_train_batches + 1
        train_generator = generate_sequences(n_train_batches, images, labels, mean, train_idxs)

        # validation sample generator
        n_validation_batches = len(validation_idxs) // batch_size
        n_remainder = len(validation_idxs) % batch_size
        if n_remainder:
            n_validation_batches = n_validation_batches + 1
        validation_generator = generate_sequences(n_validation_batches, images, labels, mean,validation_idxs)
        
        ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10,
                 verbose=1, mode='min', epsilon=1e-4, cooldown=0, min_lr=0)
        
        Checkpoint = ModelCheckpoint(filepath='/home/liuhuihui/ME/pagerank/model/resnet_aes5.h5', monitor='val_loss', verbose=1,
                 save_best_only=True, save_weights_only=False,mode='min', period=1)
        
        earlyStop = EarlyStopping(monitor='val_loss',patience=15,verbose=1,mode='min')
        
        history = model.fit_generator(generator = train_generator,
                            validation_data = validation_generator,
                            steps_per_epoch = n_train_batches,
                            validation_steps = n_validation_batches,
                            epochs = nb_epoch,
                            callbacks=[ReduceLR,Checkpoint,earlyStop])
    
    
    #path_test='/home/liuhuihui/ME/newData/ava_delta0/test_ava.hdf5'
    path_test='/home/liuhuihui/ME/data/ava/test_data.hdf5'
    with h5py.File(path_test, 'r') as test_file:
        images = test_file['images']
        labels = test_file['labels'][...]
        idxs = range(len(images))
        test_idxs = idxs[: int(len(images))]
                         
        n_test_batches = len(test_idxs) // batch_size
        n_remainder = len(test_idxs) % batch_size
        if n_remainder:
            n_test_batches = n_test_batches + 1
        test_generator = generate_sequences(n_test_batches, images, labels, mean, test_idxs)        
        predicted = model.predict_generator(generator=test_generator,steps=n_test_batches,workers=1,verbose=1)  
        print predicted      
        
        score=model.evaluate_generator(generator=test_generator,steps=n_test_batches,workers=1)
        print score
        score_ava=predicted[:,1]    
        auc_avatest=roc_auc_score(labels,score_ava)
        print 'auc_avatest',auc_avatest
        
#        sio.savemat('/home/liuhuihui/ME/pagerank/result/resnet_aes_ava.mat',
#                    {'scores':score_ava,'loss acc':score,'auc_ava':auc_avatest})

    
    path_test='/home/liuhuihui/ME/pagerank/test_data5.hdf5'
    with h5py.File(path_test, 'r') as test_file:
        images_pq = test_file['images']
        labels_pq = test_file['labels'][...]
        idxs_pq = range(len(images_pq))
        test_idxs_pq = idxs_pq[: int(len(images_pq))]
                         
        n_test_batches = len(test_idxs_pq) // batch_size
        n_remainder = len(test_idxs_pq) % batch_size
        if n_remainder:
            n_test_batches = n_test_batches + 1
        test_generator = generate_sequences(n_test_batches, images_pq, labels_pq, mean, test_idxs_pq)        
        predicted_pq = model.predict_generator(generator=test_generator,steps=n_test_batches,workers=1,verbose=1)  
        print predicted_pq      
        score2=model.evaluate_generator(generator=test_generator,steps=n_test_batches,workers=1)
        print score2
        scores_pq=predicted_pq[:,1]
        auc_pqtest=roc_auc_score(labels_pq,scores_pq)
        print 'auc_pqtest',auc_pqtest
        
#        sio.savemat('/home/liuhuihui/ME/pagerank/result/resnet_aes_pq.mat',
#                    {'scores':scores_pq,'loss acc':score2,'auc_pq':auc_pqtest})