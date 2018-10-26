#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:05:58 2017

@author: liuhuihui
"""
import numpy as np
import h5py
import scipy.io as sio
import cPickle as cp
from collections import defaultdict
import tensorflow as tf
#tf.contrib.util.make_ndarray
a512=sio.loadmat('/home/liuhuihui/ME/SLIM/result/SLIM_512_avatest_Lu.mat')
a128=sio.loadmat('/home/liuhuihui/ME/SLIM/result/SLIM_128_avatest_Lu.mat')
a256=sio.loadmat('/home/liuhuihui/ME/SLIM/result/SLIM_256_avatest_Lu.mat')
'''
path_test='/home/liuhuihui/ME/newData/CUHKPQtest/test_data5.hdf5'
with h5py.File(path_test, 'r') as test_file:
    labels_pq = test_file['labels'][...]
sio.savemat('/home/liuhuihui/labels.mat',{'labels':labels_pq})
'''
'''
score=np.load('/home/liuhuihui/DMA_Net/data_avatest/predicted_dma.npy')
score_dcnn=score[:,1]
sio.savemat('/home/liuhuihui/DMA_Net/data_avatest/dma.mat',{'scores':score_dcnn})
print score_dcnn.shape
'''





'''
path_test='/home/liuhuihui/DMA_Net/Warp/test_data.hdf5'
with h5py.File(path_test, 'r') as test_file:
    labels = test_file['labels'][...]
sio.savemat('/home/liuhuihui/ME/pagerank/labels_ava.mat',{'labels':labels})


path_test='/home/liuhuihui/ME/newData/CUHKPQtest/test_data5.hdf5'
with h5py.File(path_test, 'r') as test_file:
    labels_cuhkpq = test_file['labels'][...]
    
labels_pq=np.array(()) 
for i in range(8842):
    label= labels_cuhkpq[i][0] 
    labels_pq=np.append(labels_pq,label)
    labels_pq=np.int64(labels_pq)
sio.savemat('/home/liuhuihui/ME/pagerank/labels_pq.mat',{'labels':labels_pq})


tag_feature=np.load('/home/liuhuihui/ME/newData/feature_origin/tag_features_31-2000.npy')
user_feature=np.load('/home/liuhuihui/ME/newData/feature_origin/user_features_151-2000.npy')
group_feature=np.load('/home/liuhuihui/ME/newData/feature_origin/group_features_40-2000.npy')


#a=np.load('/home/liuhuihui/ME/newData/feature_clustered/tagFeature.npy')
#print a.shape
index=[i for i in range(4)]
print index
a=np.array([[3,4,5],[4,5,6],[5,7,8],[5,7,8]])
label=np.array([1,0,0,1])
print a
print label

random.shuffle(index)
print index
a=a[index]
label=label[index]
print a
print label
#b=a[2;0;1;3]
#print b
#path_train = '/home/liuhuihui/DMA_Net/Warp/train_data.hdf5'
#with h5py.File(path_train, 'r') as train_file:
#    images = train_file['images']
#    std=np.std(images,axis=0)
#    print std
#    data=np.asarray(images)
#    print type(data)
#    print images.shape

#a=np.array([[3,4,5,5],[3,1,5,6],[4,2,2,9]])
#a=a.astype(np.float64)
#higt,width=a.shape
#sum_a=np.sum(a,axis=1)
#
#for i in range(higt):
#    a[i,:]=a[i,:]/sum_a[i]
#    
#print a
#print a.shape
'''


#if __name__ == '__main__':
#
#    dim_tag=399   
#    input_shape = (224, 224, 3)
#    resnet50 = ResNet50(include_top=False, weights=None)
#    resnet50.load_weights('/home/liuhuihui/ME/model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',by_name=True)
#    print resnet50.output_shape
#    print resnet50.output_shape[1:]  


#a=sio.loadmat('/home/liuhuihui/DMA_Net/data_avatest/predicted_dcnn.mat')
#print a

#idxs = range(100)
#train_idxs = idxs[20: 40]
#print train_idxs
#a=np.array([[3,4,5,5],[3,1,5,6],[4,2,2,9]])
#print a
#print a[1:]
#c=[0,1]
#b=a[c,:]
#labels=np.array([[2,5,1,8,9]])
#print labels
#print labels.shape
#labels=labels[np.newaxis,...]
#print labels
#print labels.shape
#print b
#a=np.array([[3,4,5,5],[3,4,5,6]])
#b=np.array([[3,2,1,5],[5,2,10,6]])
#d=np.array([[1,2,7,5],[8,2,1,6]])
#c=np.concatenate((a,b,d),axis=1)
#print c
#print c.shape


#a=np.array([0,1,0,1,1])
#b=np.array([1,1,0,0,1])
#print type(a)
#print a.shape
#print a & b
#print a | b

#a=np.array([[2,3,4],[4,5,6]])
#print a
#print a[:,0]
#print a[:,1]
#print a[:,2]

#a=np.array((0,0,3,0))
#print np.any(a)

#a=np.array([[2,3,4],[9,5,6],[6,7,8],[8,9,10]])
#print a
#print a[:,1:]
##print a[::,::(-1)]
#print a[:,:2]

#b=a[:-(5):-1]
#print b
#print np.argsort(a,axis=1)
#top_k_indx=np.argsort(a,axis=1)[:,1:]
#print top_k_indx
#a=np.array([2,3,4,6,7,8])
#print a
#print a.shape
#print a[:4]


'''
feature=np.load('/home/liuhuihui/ME/newData/tag_features_31-2000.npy')
feature=np.int64(feature)

[img,num]=feature.shape

matrix=np.transpose(feature)
simiMatr=np.zeros((num,num))
a=matrix[0,:]
b=matrix[1,:]

a=np.asarray(a)
b=np.asarray(b)
print a & b
inter=a & b
union=a | b
num_union=np.sum(union)
num_inter=np.sum(inter)
if num_union != 0:
    simi=num_inter/float(num_union)
    print simi
'''