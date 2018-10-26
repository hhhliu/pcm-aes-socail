# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:38:11 2018

@author: liuhuihui
"""

import numpy as np
from sklearn.cluster import AffinityPropagation
import cPickle

#fourth
#399  
score_pageRank_dict = cPickle.load(open('/home/liuhuihui/ME/pagerank/tag/data/score_pageRank_dict.pkl','r'))

#50000*2210
tag_feature=np.load('/home/liuhuihui/ME/newData/feature_origin/tag_features_31-2000.npy')
#50000*399
tagFeature_01=np.load('/home/liuhuihui/ME/newData/feature_01/tagFeature.npy')
#2210*1
labels=np.load('/home/liuhuihui/ME/pagerank/tag/data/af_labels.npy')

def cal_pagerank(indice_tags,indice_clusters):
    print indice_tags
    print indice_clusters
    lens_tag=len(indice_tags)
    lens_cluster=len(indice_clusters)
    tagscore=np.zeros((399))    
    
    for i in range(lens_cluster):
        cluster=indice_clusters[i]
        #dict
        score_pagerank=score_pageRank_dict[cluster]
        sum_score=0
        n=0
        for j in range(lens_tag):
            tag=indice_tags[j]
            
            if labels[tag]==cluster:
                score=score_pagerank[tag]                               
                sum_score+=score
                n+=1
        if n!=0:
            mean_score=sum_score/n
            tagscore[cluster]=mean_score
    return tagscore
     
tagFeature_score_pG=np.zeros((50000,399))
for i in range(50000):
    print i
    tagFeature_2210=tag_feature[i]   #2210
    indice_tag_nonzeros=tagFeature_2210.nonzero()

    cluster_indice = tagFeature_01[i]   #399
    indice_cluster_nonzeros=cluster_indice.nonzero()
    
    tagscore = cal_pagerank(indice_tag_nonzeros[0],indice_cluster_nonzeros[0])
    tagFeature_score_pG[i]=tagscore     
    
np.save('/home/liuhuihui/ME/pagerank/tag/data/tagFeature_score_pG.npy',tagFeature_score_pG) 