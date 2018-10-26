# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:36:50 2018

@author: liuhuihui
"""

import numpy as np
from sklearn.cluster import AffinityPropagation
import cPickle
from ..pagerank import pageRank

tag_indices_dict = cPickle.load(open('/home/liuhuihui/ME/pagerank/tag/data/tag_indices_dict.pkl','rb'))
tag_simiMatr_dict = cPickle.load(open('/home/liuhuihui/ME/pagerank/tag/data/tag_simiMatr_dict.pkl','rb'))
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
    
cPickle.dump(score_pageRank_dict,open('/home/liuhuihui/ME/pagerank/tag/data/score_pageRank_dict.pkl','wb'))