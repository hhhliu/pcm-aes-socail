# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 21:19:20 2017

@author: Administrator
"""

"""
Load images and labels, and save them into one single h5 format file.
"""

import numpy as np
import h5py
import random
from keras.preprocessing import image
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

image_dir = '/home/liuhuihui/AVA/'
label_path = '/home/liuhuihui/AVAset/labels.txt'

indices, labels = [], []
n_high=0
n_low=0
with open(label_path, 'rb') as f:
    for l in f:
        l = l.strip().split()
        print l[0]
        sum_people=0
        sum_score=0
        k=1
        for j in range(2,12):
            x=int(l[j])
            sum_people+=x
            sum_score+=x*k
            k+=1
        sum_people=float(sum_people)
        avg=sum_score/sum_people
        if avg<5:
            labels.append(0)
            indices.append(l[0])
            n_low+=1
        elif avg>=5:
            labels.append(1)
            indices.append(l[0])
            n_high+=1
            
labels = np.asarray(labels)
print labels.shape
print n_high
print n_low
# random permutation
perm = range(len(indices))
random.seed(702)
random.shuffle(perm)
indices = [indices[i] for i in perm]
labels = labels[perm]

# divide the samples into 80% train and 20% test
train_indices = indices[:int(0.8*len(indices))]
train_labels = labels[:int(0.8*len(labels))]

test_indices = indices[int(0.8*len(indices)):]
test_labels = labels[int(0.8*len(labels)):]

'''
np.save('/home/liuhuihui/ME/data/train_indices.npy',train_indices)
np.save('/home/liuhuihui/ME/data/train_labels.npy',train_labels)
np.save('/home/liuhuihui/ME/data/test_indices.npy',test_indices)
np.save('/home/liuhuihui/ME/data/test_labels.npy',test_labels)
'''

# channel last format following Tensorflow
train_shape = (len(train_indices), 256, 256, 3)
test_shape = (len(test_indices), 256, 256, 3)

mean = np.zeros(train_shape[1:], np.float32)
n_images = float(len(train_indices) + len(test_indices))

# creat training data and save images
train_file = h5py.File('/home/liuhuihui/ME/newData/ava_delta0/train_ava56.hdf5', 'w')
dset = train_file.create_dataset("images", train_shape, np.uint8)   
for idx, fid in enumerate(train_indices):
    if (idx + 1) % 1000 == 0:
        print 'Train data: {}/{}'.format(idx + 1, len(train_indices))
    addr = image_dir + fid + '.jpg'
    img = image.load_img(addr,target_size=(256,256))
    x = image.img_to_array(img)
    dset[idx, ...] = x
    mean += x/n_images

# creat testing data and save images
test_file = h5py.File('/home/liuhuihui/ME/newData/ava_delta0/test_ava54.hdf5', 'w')
dset = test_file.create_dataset("images", test_shape, np.uint8)
for idx, fid in enumerate(test_indices):
    if (idx + 1) % 1000 == 0:
        print 'Test data: {}/{}'.format(idx + 1, len(test_indices))
    addr = image_dir + fid + '.jpg'
    img = image.load_img(addr,target_size=(256,256))
    x = image.img_to_array(img)
    dset[idx, ...] = x

# save labels
train_file.create_dataset("labels", data = train_labels)
test_file.create_dataset("labels", data = test_labels)

# save mean value
train_file.create_dataset("mean", data = mean)
test_file.create_dataset("mean", data = mean)

train_file.close()
test_file.close()
