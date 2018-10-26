#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 15:55:14 2017

@author: liuhuihui
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 09:54:39 2017

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
from keras.layers import Input, Flatten, Dense
import matplotlib.pyplot as plt
import h5py
import random
from keras import losses,metrics
import scipy.io as sio
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import decode_predictions
from keras.models import Model
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb
import keras.backend as K
#from tensorflow.python.keras.__impl.keras import backend as K
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

def generate_sequences(n_batches, images, labels, mean, idxs):
    
    while True:
        for bid in xrange(0, n_batches):
            if bid == n_batches - 1:
                batch_idxs = idxs[bid * batch_size:]
            else:
                batch_idxs = idxs[bid * batch_size: (bid + 1) * batch_size]
                                  
            batch_length=len(batch_idxs)
            X = np.zeros((batch_length,224,224,3),np.float32)            
            Y = labels[batch_idxs,:]          
            
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
            
            yield (X, Y)
            
               
if __name__ == '__main__':
    
    input_shape = (224, 224, 3)
    input_tensor = Input(shape=input_shape)
    resnet50 = ResNet50(include_top=False, weights=None,input_tensor=input_tensor)
    resnet50.load_weights('/home/liuhuihui/ME/model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',by_name=True)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=resnet50.output_shape[1:]))
    top_model.add(Dense(399, activation='sigmoid'))
    model = Model(input=resnet50.input, output=top_model(resnet50.output))
    for layer in model.layers[:174]:
        layer.trainable=False    
    
    sgd = SGD(lr=0.0001, momentum=0.9, decay=1e-6, nesterov=True)
    
    def weighted_binary_crossentropy(target,output):
        POS_WEIGHT=10
        _epsilon=tfb._to_tensor(tfb.epsilon(),output.dtype.base_dtype)
        output=tf.clip_by_value(output,_epsilon,1-_epsilon)
        output=tf.log(output/(1-output))    
        loss=tf.nn.weighted_cross_entropy_with_logits(targets=target,logits=output,pos_weight=POS_WEIGHT)       
        return tf.reduce_mean(loss,axis=-1) 
           
    def my_acc(y_true,y_pred):
        correct_prediction = tf.equal(tf.round(y_pred), y_true)        
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return evaluation_step
        
    #model.compile(optimizer=sgd,loss='kullback_leibler_divergence')
    model.compile(optimizer=sgd,loss='categorical_crossentropy')
    #model.compile(optimizer=sgd,loss=weighted_binary_crossentropy)
    model.summary()

    batch_size = 64
    nb_epoch = 200
    validation_ratio = 0.1
    
    path_train = '/home/liuhuihui/ME/newData/tag/train_flickr.hdf5'
    with h5py.File(path_train, 'r') as train_file:
        images = train_file['images']      
        mean = train_file['mean'][...]
        labels = train_file['labels_tag_01'][...]
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
        validation_generator = generate_sequences(n_validation_batches, images, labels, mean, validation_idxs)

        ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10,
                 verbose=1, mode='min', epsilon=1e-4, cooldown=0, min_lr=0)
        
        Checkpoint = ModelCheckpoint(filepath='/home/liuhuihui/ME/newData/model/model-finetune-ResNet50-tag-01-weighted.h5', monitor='val_loss', verbose=1,
                 save_best_only=True, save_weights_only=False,mode='min', period=1)
        
        earlyStop = EarlyStopping(monitor='val_loss',patience=15,verbose=1,mode='min')
        
        history = model.fit_generator(generator = train_generator,
                            validation_data = validation_generator,
                            steps_per_epoch = n_train_batches,
                            validation_steps = n_validation_batches,
                            epochs = nb_epoch,
                            callbacks=[ReduceLR,Checkpoint,earlyStop])


    path_test='/home/liuhuihui/ME/newData/tag/test_flickr.hdf5'
    with h5py.File(path_test, 'r') as test_file:
        images = test_file['images']
        mean = test_file['mean'][...]
        labels = test_file['labels_tag_01'][...]
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