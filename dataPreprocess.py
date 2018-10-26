# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:15:01 2018

@author: liuhuihui
"""

import numpy as np
from sklearn.cluster import AffinityPropagation
import cPickle
from pagerank import pageRank


#fourth
#399  
score_pageRank_dict = cPickle.load(open('/home/liuhuihui/ME/pagerank/data/score_pageRank_dict.pkl','r'))

#50000*2210
tag_feature=np.load('/home/liuhuihui/ME/newData/feature_origin/tag_features_31-2000.npy')
#50000*399
tagFeature_01=np.load('/home/liuhuihui/ME/newData/feature_01/tagFeature.npy')
#2210*1
labels=np.load('/home/liuhuihui/ME/pagerank/data/af_labels.npy')

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
    
np.save('/home/liuhuihui/ME/pagerank/data/tagFeature_score_pG.npy',tagFeature_score_pG)  


'''
#third: pagerank--  : score_pageRank_dict

tag_indices_dict = cPickle.load(open('/home/liuhuihui/ME/pagerank/data/tag_indices_dict.pkl','rb'))
tag_simiMatr_dict = cPickle.load(open('/home/liuhuihui/ME/pagerank/data/tag_simiMatr_dict.pkl','rb'))
score_pageRank_dict={}
for i in range(399):
    tag_indices=tag_indices_dict[i]
    tag_simiMatr=tag_simiMatr_dict[i]
    num_tag=len(tag_indices)
    score_pageRank={}
    #score_pG is a list, and sum is 1
    score_pG=pageRank(tag_simiMatr)
    for j in range(num_tag):
        tag=tag_indices[j]
        score=score_pG[j]
        score_pageRank[tag]=score
    score_pageRank_dict[i]=score_pageRank
    
cPickle.dump(score_pageRank_dict,open('/home/liuhuihui/ME/pagerank/data/score_pageRank_dict.pkl','wb'))
'''
      

'''
#sencond: tag_simiMatr_dict

tag_indices_dict = cPickle.load(open('/home/liuhuihui/ME/pagerank/data/tag_indices_dict.pkl','rb'))
simiMatr_tag=np.load('/home/liuhuihui/ME/newData/simi/simiMatr_tag.npy')

def cal_simi(tag_indices):
    lens=len(tag_indices)
    tag_simiMatr=np.zeros((lens,lens))
    for m in range(lens):
        indice_i=tag_indices[m]
        for n in range(lens):            
            indice_j=tag_indices[n]
            tag_simiMatr[m][n]=simiMatr_tag[indice_i][indice_j]
    return tag_simiMatr

tag_simiMatr_dict={}
for i in range(399):
    tag_indices=tag_indices_dict[i]
    tag_simiMatr=cal_simi(tag_indices)    
    tag_simiMatr_dict[i]=tag_simiMatr
cPickle.dump(tag_simiMatr_dict,open('/home/liuhuihui/ME/pagerank/data/tag_simiMatr_dict.pkl','wb'))




#first: tag_indices_dict
simiMatr_tag=np.load('/home/liuhuihui/ME/newData/simi/simiMatr_tag.npy')
af = AffinityPropagation(affinity= 'precomputed').fit(simiMatr_tag)
cluster_centers_indices=af.cluster_centers_indices_
labels=af.labels_
affinity_matrix=af.affinity_matrix_
n_iter=af.n_iter_

np.save('/home/liuhuihui/ME/pagerank/data/cluster_centers_indices.npy',cluster_centers_indices)
np.save('/home/liuhuihui/ME/pagerank/data/af_labels.npy',labels)

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
cPickle.dump(tag_indices_dict,open('/home/liuhuihui/ME/pagerank/data/tag_indices_dict.pkl','wb'))
print tag_indices_dict

'''
