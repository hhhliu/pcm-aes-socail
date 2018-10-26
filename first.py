# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:32:20 2018

@author: liuhuihui
"""
import numpy as np
from sklearn.cluster import AffinityPropagation
import cPickle

simiMatr_tag=np.load('/home/liuhuihui/ME/newData/simi/simiMatr_tag.npy')
af = AffinityPropagation(affinity= 'precomputed').fit(simiMatr_tag)
cluster_centers_indices=af.cluster_centers_indices_
labels=af.labels_
affinity_matrix=af.affinity_matrix_
n_iter=af.n_iter_

np.save('/home/liuhuihui/ME/pagerank/tag/data/cluster_centers_indices.npy',cluster_centers_indices)
np.save('/home/liuhuihui/ME/pagerank/tag/data/af_labels.npy',labels)

tag_indices_dict={}
for i in range(399):
#    cci = cluster_centers_indices[i]
#    print cci
    tag_indices=[]
    for j in range(2210):
        if labels[j]==i:
            tag_indices.append(j)
        
    tag_indices_dict[i]=tag_indices 
    
#save
cPickle.dump(tag_indices_dict,open('/home/liuhuihui/ME/pagerank/tag/data/tag_indices_dict.pkl','wb'))
print tag_indices_dict