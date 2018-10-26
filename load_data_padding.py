#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 22:49:59 2017

@author: liuhuihui
"""

import numpy as np
import h5py
from keras.preprocessing import image
from PIL import Image
from keras.applications.resnet50 import preprocess_input
outsize=256
    
def padding(path):
    img=image.load_img(path) 
    new_img=Image.new('RGB',(256,256))
    (width,height)=img.size
    if width<height:
        new_height=256
        new_width=256*width/height
        im=img.resize((new_width,new_height))
        left=(256-new_width)/2
        right=left+new_width
        new_img.paste(im,(left,0,right,256))      
    elif width>height:
        new_width=256
        new_height=256*height/width
        im=img.resize((new_width,new_height))
        upper=(256-new_height)/2
        lower=upper+new_height
        new_img.paste(im,(0,upper,256,lower))       
    else:
        new_img=img.resize((outsize,outsize))
        
    new_img=np.asarray(new_img)
    return new_img 
    
image_dir = '/home/liuhuihui/Data/AVA/'
label_path = '/home/liuhuihui/demo_cui/Dataset/AVA/labels.txt'

indices=np.load('/home/liuhuihui/RAPID_Lu/dataSet/indices.npy')
labels=np.load('/home/liuhuihui/RAPID_Lu/dataSet/labels.npy')

# divide the samples into 80% train and 20% test
train_indices = indices[:int(0.8*len(indices))]
train_labels = labels[:int(0.8*len(labels))]

test_indices = indices[int(0.8*len(indices)):]
test_labels = labels[int(0.8*len(labels)):]

# channel last format following Tensorflow
train_shape = (len(train_indices), 256, 256, 3)
test_shape = (len(test_indices), 256, 256, 3)

mean = np.zeros(train_shape[1:], np.float32)
n_images = float(len(train_indices) + len(test_indices))

# creat training data and save images
train_file = h5py.File('/home/liuhuihui/RAPID_Lu/dataSet/Padding/train_data.hdf5', 'w')
dset = train_file.create_dataset("images", train_shape, np.uint8)
for idx, fid in enumerate(train_indices):
    if (idx + 1) % 1000 == 0:
        print 'Train data: {}/{}'.format(idx + 1, len(train_indices))
    addr = image_dir + fid + '.jpg'
    x = padding(addr)
    dset[idx, ...] = x
    mean += x/n_images

# creat testing data and save images
test_file = h5py.File('/home/liuhuihui/RAPID_Lu/dataSet/Padding/test_data.hdf5', 'w')
dset = test_file.create_dataset("images", test_shape, np.uint8)
for idx, fid in enumerate(test_indices):
	if (idx + 1) % 1000 == 0:
		 print 'Test data: {}/{}'.format(idx + 1, len(test_indices))

	addr = image_dir + fid + '.jpg'
	x = padding(addr)
	dset[idx, ...] = x
	mean += x / n_images

# save labels
train_file.create_dataset("labels", data = train_labels)
test_file.create_dataset("labels", data = test_labels)

# save mean value
train_file.create_dataset("mean", data = mean)
test_file.create_dataset("mean", data = mean)

train_file.close()
test_file.close()