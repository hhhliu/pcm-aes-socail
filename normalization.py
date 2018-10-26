#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 10:34:45 2017

@author: liuhuihui
"""

import numpy as np

'''
a=np.load('/home/liuhuihui/ME/pagerank/tag/data/tagFeature_score_pG_norm.npy')
sum_a=np.sum(a,axis=1)
'''
'''
#50000,399
tagFeature=np.load('/home/liuhuihui/ME/pagerank/tag/data/tagFeature_score_pG.npy')
tagFeature=tagFeature.astype(np.float64)
sum_tag=np.sum(tagFeature,axis=1)
higt,width=tagFeature.shape
for i in range(higt):
    print i
    if sum_tag[i]!=0:
        tagFeature[i,:]=tagFeature[i,:]/sum_tag[i]
np.save('/home/liuhuihui/ME/pagerank/tag/data/tagFeature_score_pG_norm.npy',tagFeature)

'''
'''
userFeature=np.load('/home/liuhuihui/ME/pagerank/user/data/userFeature_score_pG.npy')
userFeature=userFeature.astype(np.float64)
sum_user=np.sum(userFeature,axis=1)
higt,width=userFeature.shape
for i in range(higt):
    print i
    if sum_user[i]!=0:
        userFeature[i,:]=userFeature[i,:]/sum_user[i]
np.save('/home/liuhuihui/ME/pagerank/user/data/userFeature_score_pG_norm.npy',userFeature)


'''
groupFeature=np.load('/home/liuhuihui/ME/pagerank/group/data/groupFeature_score_pG.npy')
groupFeature=groupFeature.astype(np.float64)
sum_group=np.sum(groupFeature,axis=1)
higt,width=groupFeature.shape
for i in range(higt):
    print i
    if sum_group[i]!=0:
        groupFeature[i,:]=groupFeature[i,:]/sum_group[i]
np.save('/home/liuhuihui/ME/pagerank/group/data/groupFeature_score_pG_norm.npy',groupFeature)

sum_group=np.sum(groupFeature,axis=1)
