#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 19:46:57 2018

@author: liuhuihui
"""

import numpy as np
import cPickle as cp
from keras.preprocessing import image
import h5py
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

outsize=256
n=50000.0
shape=(50000,256,256,3)
mean=np.zeros((256,256,3),np.float32)
labels_tag = np.load('/home/liuhuihui/ME/pagerank/tag/data/tagFeature_score_pG_norm.npy')
labels_user= np.load('/home/liuhuihui/ME/pagerank/user/data/userFeature_score_pG_norm.npy')
labels_group = np.load('/home/liuhuihui/ME/pagerank/group/data/groupFeature_score_pG_norm.npy')

nameph='/home/liuhuihui/ME/tag/data/flickr50000_tag.cp'
namefile=open(nameph,'rb')
id_tags=cp.load(namefile) 
namefile.close()

indices=[]
for k,v in id_tags.iteritems():
    indices.append(k)

data_flickr=h5py.File('/home/liuhuihui/ME/pagerank/data_flickr.hdf5','w')
dset_train=data_flickr.create_dataset("images",shape,np.uint8)

root='/home/liuhuihui/now_work/dataset/Flickr/photos/'
for i in range(50000):
    path=root+indices[i]+'.jpg'
    print path
    img=image.load_img(path,target_size=(256,256))
    x=image.img_to_array(img)
    dset_train[i,...]=x
    mean += x/n

data_flickr.create_dataset("labels_tag",data=labels_tag)
data_flickr.create_dataset("labels_user",data=labels_user)
data_flickr.create_dataset("labels_group",data=labels_group)
data_flickr.create_dataset("mean",data=mean)
data_flickr.close()

