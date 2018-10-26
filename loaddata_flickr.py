#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:34:46 2017

@author: liuhuihui
"""

import numpy as np
import os
import cPickle as cp
from keras.preprocessing import image
import h5py
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

outsize=256
n_train=40000.0
train_shape=(40000,256,256,3)
test_shape=(10000,256,256,3)
mean=np.zeros((256,256,3),np.float32)

labels_tag_norm=np.load('/home/liuhuihui/ME/newData/feature_normalization/tagFeature.npy')
labels_tag_01=np.load('/home/liuhuihui/ME/newData/feature_01/tagFeature.npy')
train_taglabels_norm=labels_tag_norm[:40000]
test_taglabels_norm=labels_tag_norm[40000:]
train_taglabels_01=labels_tag_01[:40000]
test_taglabels_01=labels_tag_01[40000:]

labels_user_norm=np.load('/home/liuhuihui/ME/newData/feature_normalization/userFeature.npy')
labels_user_01=np.load('/home/liuhuihui/ME/newData/feature_01/userFeature.npy')
train_userlabels_norm=labels_user_norm[:40000]
test_userlabels_norm=labels_user_norm[40000:]
train_userlabels_01=labels_user_01[:40000]
test_userlabels_01=labels_user_01[40000:]

labels_group_norm=np.load('/home/liuhuihui/ME/newData/feature_normalization/groupFeature.npy')
labels_group_01=np.load('/home/liuhuihui/ME/newData/feature_01/groupFeature.npy')
train_grouplabels_norm=labels_group_norm[:40000]
test_grouplabels_norm=labels_group_norm[40000:]
train_grouplabels_01=labels_group_01[:40000]
test_grouplabels_01=labels_group_01[40000:]

nameph='/home/liuhuihui/ME/tag/data/flickr50000_tag.cp'
namefile=open(nameph,'rb')
id_tags=cp.load(namefile) 
namefile.close()

indices=[]
for k,v in id_tags.iteritems():
    indices.append(k)

train_indices=indices[:40000]
test_indices=indices[40000:]

train_flickr=h5py.File('/home/liuhuihui/ME/newData/train_flickr.hdf5','w')
dset_train=train_flickr.create_dataset("images",train_shape,np.uint8)

root='/home/liuhuihui/now_work/dataset/Flickr/photos/'
for i in range(40000):
    path=root+train_indices[i]+'.jpg'
    print path
    img=image.load_img(path,target_size=(256,256))
    x=image.img_to_array(img)
    dset_train[i,...]=x
    mean += x/n_train

train_flickr.create_dataset("labels_tag_norm",data=train_taglabels_norm)
train_flickr.create_dataset("labels_tag_01",data=train_taglabels_01)
train_flickr.create_dataset("labels_user_norm",data=train_userlabels_norm)
train_flickr.create_dataset("labels_user_01",data=train_userlabels_01)
train_flickr.create_dataset("labels_group_norm",data=train_grouplabels_norm)
train_flickr.create_dataset("labels_group_01",data=train_grouplabels_01)
train_flickr.create_dataset("mean",data=mean)
train_flickr.close()

test_flickr=h5py.File('/home/liuhuihui/ME/newData/test_flickr.hdf5','w')
dset_test=test_flickr.create_dataset("images",test_shape,np.uint8)

for i in range(10000):
    path=root+test_indices[i]+'.jpg'
    print path
    img=image.load_img(path,target_size=(256,256))
    x=image.img_to_array(img)
    dset_test[i,...]=x
    
test_flickr.create_dataset("labels_tag_norm",data=test_taglabels_norm)
test_flickr.create_dataset("labels_tag_01",data=test_taglabels_01)
test_flickr.create_dataset("labels_user_norm",data=test_userlabels_norm)
test_flickr.create_dataset("labels_user_01",data=test_userlabels_01)
test_flickr.create_dataset("labels_group_norm",data=test_grouplabels_norm)
test_flickr.create_dataset("labels_group_01",data=test_grouplabels_01)
test_flickr.create_dataset("mean",data=mean)
test_flickr.close()
