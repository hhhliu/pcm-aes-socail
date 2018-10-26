#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:10:43 2017

@author: liuhuihui
"""
import numpy as np
import cPickle as cp
import h5py
import scipy.io as sio

tag_feature=np.load('/home/liuhuihui/ME/data/tag_features.npy')
print tag_feature
print type(tag_feature)
print tag_feature.shape


sio.savemat('/home/liuhuihui/ME/data/tag_feature.mat',{'tag_feature':tag_feature})