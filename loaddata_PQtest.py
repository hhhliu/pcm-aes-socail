# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:24:35 2018

@author: liuhuihui
"""


import numpy as np
import h5py
import random
from keras.preprocessing import image
import os

outsize=256  
image_testhigh_dir = '/home/liuhuihui/now_work/dataset/CUHKPQ/test_high/'
image_testlow_dir = '/home/liuhuihui/now_work/dataset/CUHKPQ/test_low/'  

n_testhigh=2262
n_testlow=6580
n_test=n_testhigh+n_testlow
labels_testhigh=np.ones((n_testhigh,1))
labels_testlow=np.zeros((n_testlow,1))
labels_test=np.concatenate((labels_testhigh,labels_testlow),axis=0)

testhigh_shape = (n_testhigh, 256, 256, 3)
testlow_shape = (n_testlow, 256, 256, 3)
test_shape=(n_test,256,256,3)
dset_test =np.zeros((n_test,256,256,3),dtype=np.uint8)
idx=0
for root,dirs,files in os.walk(image_testhigh_dir):
    for file in files:
        print idx+1
        addr=os.path.join(root,file)
        img = image.load_img(addr,target_size=(256,256))
        data = image.img_to_array(img) 
        dset_test[idx,:] = data 
        idx+=1
    
for root,dirs,files in os.walk(image_testlow_dir):
    for file in files:
        print idx+1
        addr=os.path.join(root,file)
        img = image.load_img(addr,target_size=(256,256))
        data = image.img_to_array(img) 
        dset_test[idx,:] = data 
        idx+=1
        
print dset_test.shape
index=[i for i in range(n_test)]
random.shuffle(index)
dset_test=dset_test[index]
labels_test=labels_test[index]

test_file = h5py.File('/home/liuhuihui/ME/newData/CUHKPQtest/test_data5.hdf5', 'w')
test_file.create_dataset("images", data = dset_test)
test_file.create_dataset("labels", data = labels_test)
test_file.close()



