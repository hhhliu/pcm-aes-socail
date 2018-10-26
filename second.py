# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:36:09 2018

@author: liuhuihui
"""
import numpy as np
import cPickle

tag_indices_dict = cPickle.load(open('/home/liuhuihui/ME/pagerank/tag/data/tag_indices_dict.pkl','rb'))
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
cPickle.dump(tag_simiMatr_dict,open('/home/liuhuihui/ME/pagerank/tag/data/tag_simiMatr_dict.pkl','wb'))
