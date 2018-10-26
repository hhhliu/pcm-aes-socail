#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:05:53 2018

@author: liuhuihui
"""
import h5py
path_train = '/home/liuhuihui/ME/newData/train_flickr.hdf5'
with h5py.File(path_train, 'r') as train_file:
    images = train_file['images']
    print images.shape
    
path_test='/home/liuhuihui/ME/newData/test_flickr.hdf5'
with h5py.File(path_test, 'r') as test_file:
    images = test_file['images']
    print images.shape
    
data_flickr=h5py.File('/home/liuhuihui/ME/pagerank/data_flickr.hdf5','w')
dset_train=data_flickr.create_dataset("images",shape,np.uint8)