#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 21:53:02 2017

@author: liuhuihui
"""

import numpy as np
        

def cal_simi(feature):
    feature=np.int64(feature)
    [img,num]=feature.shape
    matrix=np.transpose(feature)
    simiMatr=np.zeros((num,num))
    for i in range(num):
        for j in range(i,num):
            a=matrix[i,:]
            b=matrix[j,:]
            union=a | b
            inter=a & b
            num_union=np.sum(union)
            num_inter=np.sum(inter)
            if num_union != 0:
                simi=num_inter/float(num_union)
                simiMatr[i][j]=simi
                simiMatr[j][i]=simi
                print simi
    return simiMatr
    
    
if __name__ == '__main__':
    
    tag_feature=np.load('/home/liuhuihui/ME/newData/tag_features_31-2000.npy')
    user_feature=np.load('/home/liuhuihui/ME/newData/user_features_151-2000.npy')
    group_feature=np.load('/home/liuhuihui/ME/newData/group_features_40-2000.npy')
    
    simiMatr_tag=cal_simi(tag_feature)
    np.save('/home/liuhuihui/ME/newData/simiMatr_tag.npy',simiMatr_tag)
    simiMatr_user=cal_simi(user_feature)
    np.save('/home/liuhuihui/ME/newData/simiMatr_user.npy',simiMatr_user)
    simiMatr_group=cal_simi(group_feature)
    np.save('/home/liuhuihui/ME/newData/simiMatr_group.npy',simiMatr_group)
