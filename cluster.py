#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 10:47:01 2017

@author: liuhuihui
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import spectral_clustering
from sklearn.cluster import AffinityPropagation

'''
tag_feature=np.load('/home/liuhuihui/ME/newData/feature_origin/tag_features_31-2000.npy')
simiMatr_tag=np.load('/home/liuhuihui/ME/newData/simi/simiMatr_tag.npy')
af = AffinityPropagation(affinity= 'precomputed').fit(simiMatr_tag)
cluster_centers_indices=af.cluster_centers_indices_
#cluster_centers=af.cluster_centers_
labels=af.labels_
affinity_matrix=af.affinity_matrix_
n_iter=af.n_iter_
tagFeature=np.zeros((50000,399))
for i in range(50000):
    for j in range(2210):
        tag_label = tag_feature[i][j]
        if tag_label != 0:
            label=labels[j]
            tagFeature[i][label]=1 
            #tagFeature[i][label]+=1 

#np.save('/home/liuhuihui/ME/newData/feature_01/tagFeature.npy',tagFeature)    
'''
'''
tag_indices_dict={}
for i in range(399):
    cci = cluster_centers_indices[i]
    for j in range(2210):
        tag_indices=[]
        if labels[j]==cci:
            tag_indices.append(j)
        
        tag_indices_dict[i]=tag_indices   
'''        
        
    


'''
user_feature=np.load('/home/liuhuihui/ME/newData/feature_origin/user_features_151-2000.npy')
simiMatr_user=np.load('/home/liuhuihui/ME/newData/simi/simiMatr_user.npy')
af = AffinityPropagation(affinity= 'precomputed').fit(simiMatr_user)
cluster_centers_indices=af.cluster_centers_indices_
labels=af.labels_
affinity_matrix=af.affinity_matrix_
n_iter=af.n_iter_

userFeature=np.zeros((50000,240))
for i in range(50000):
    for j in range(5776):
        tag_label = user_feature[i][j]
        if tag_label != 0:
            label=labels[j]
            userFeature[i][label]=1      
            #userFeature[i][label]+=1          
   
np.save('/home/liuhuihui/ME/newData/feature_01/userFeature.npy',userFeature)   
'''

group_feature=np.load('/home/liuhuihui/ME/newData/feature_origin/group_features_40-2000.npy')
simiMatr_group=np.load('/home/liuhuihui/ME/newData/simi/simiMatr_group.npy')
af = AffinityPropagation(affinity= 'precomputed').fit(simiMatr_group)
cluster_centers_indices=af.cluster_centers_indices_
labels=af.labels_
affinity_matrix=af.affinity_matrix_
n_iter=af.n_iter_
'''
groupFeature=np.zeros((50000,327))
for i in range(50000):
    for j in range(2861):
        tag_label = group_feature[i][j]
        if tag_label != 0:
            label=labels[j]
            groupFeature[i][label]=1      
            #groupFeature[i][label]+=1          
   
np.save('/home/liuhuihui/ME/newData/feature_01/groupFeature.npy',groupFeature)   
'''