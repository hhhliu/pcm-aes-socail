# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:24:46 2018

@author: liuhuihui
"""

import numpy as np
import h5py
import random
from keras.preprocessing import image
from PIL import Image
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

image_dir = '/home/liuhuihui/AVA/'
label_path = '/home/liuhuihui/AVAset/labels.txt'
        
train_indices=np.load('/home/liuhuihui/ME/data/train_indices.npy')
train_labels=np.load('/home/liuhuihui/ME/data/train_labels.npy')
test_indices=np.load('/home/liuhuihui/ME/data/test_indices.npy')
test_labels=np.load('/home/liuhuihui/ME/data/test_labels.npy')

train_high=40112
train_low=32100  #6420*5
n_train=72212

test_high=10072
test_low=1561
n_test=11633
new_train_labels=[]

train_shape = (n_train, 256, 256, 3)
test_shape = (n_test, 256, 256, 3)

mean = np.zeros(train_shape[1:], np.float32)
n_images = float(n_train + n_test)

#permutation=np.random.permutation(n_train)
train_file = h5py.File('/home/liuhuihui/ME/newData/ava/train_ava.hdf5', 'w')
dset = train_file.create_dataset("images", train_shape, np.uint8)   
for idx, fid in enumerate(train_indices):
    print idx ,fid
    if (idx + 1) % 1000 == 0:
        print 'Train data: {}/{}'.format(idx + 1, len(train_indices))   
    addr = image_dir + fid + '.jpg'
    if train_labels[idx]==1:
        new_train_labels.append(1)
        img = image.load_img(addr,target_size=(256,256))
        x = image.img_to_array(img)
        dset[idx, ...] = x
        mean += x/n_images
    else:
        img = image.load_img(addr,target_size=(256,256))
        x = image.img_to_array(img)
        dset[idx, ...] = x
        mean += x/n_images
        new_train_labels.append(0)

        im=Image.open(addr)
        
        out1=im.transpose(Image.FLIP_LEFT_RIGHT)
        addr1=r'/home/liuhuihui/ME/images_DA/'+fid+'_lr.jpg'
        out1.save(addr1)
        img = image.load_img(addr1,target_size=(256,256))
        x = image.img_to_array(img)
        dset[idx, ...] = x
        mean += x/n_images
        new_train_labels.append(0)

        out2=im.transpose(Image.FLIP_TOP_BOTTOM)
        addr2=r'/home/liuhuihui/ME/images_DA/'+fid+'_tb.jpg'
        out2.save(addr2)
        img = image.load_img(addr2,target_size=(256,256))
        x = image.img_to_array(img)
        dset[idx, ...] = x
        mean += x/n_images
        new_train_labels.append(0)

        out3=im.transpose(Image.ROTATE_180)
        addr3=r'/home/liuhuihui/ME/images_DA/'+fid+'_180.jpg'
        out3.save(addr3)
        img = image.load_img(addr3,target_size=(256,256))
        x = image.img_to_array(img)
        dset[idx, ...] = x
        mean += x/n_images
        new_train_labels.append(0)

        out4=im.transpose(Image.ROTATE_90)   #ni shi zhen rotate
        addr4=r'/home/liuhuihui/ME/images_DA/'+fid+'_90.jpg'
        out4.save(addr4)
        img = image.load_img(addr4,target_size=(256,256))
        x = image.img_to_array(img)
        dset[idx, ...] = x
        mean += x/n_images
        new_train_labels.append(0)
        
#dset_new=dset[permutation,...]
#train_labels=new_train_labels[permutation]

test_file = h5py.File('/home/liuhuihui/ME/newData/ava/test_ava.hdf5', 'w')
dset = test_file.create_dataset("images", test_shape, np.uint8)
for idx, fid in enumerate(test_indices):
    if (idx + 1) % 1000 == 0:
        print 'Test data: {}/{}'.format(idx + 1, len(test_indices))
    addr = image_dir + fid + '.jpg'
    img = image.load_img(addr,target_size=(256,256))
    x = image.img_to_array(img)
    dset[idx, ...] = x

# save labels
train_file.create_dataset("labels", data = new_train_labels)
test_file.create_dataset("labels", data = test_labels)

# save mean value
train_file.create_dataset("mean", data = mean)
test_file.create_dataset("mean", data = mean)

train_file.close()
test_file.close()
