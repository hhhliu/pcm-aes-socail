#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 20:46:16 2018

@author: liuhuihui
"""


from keras import backend as K
from keras.objectives import categorical_crossentropy

def tag_loss(a):
	def tag_losses(y_true, y_pred):
         loss_a=categorical_crossentropy(y_true, y_pred)
         loss_ab=abs(loss_a)          
         tag_ls=a*K.sum(loss_ab)
         return tag_ls
	return tag_losses


def user_loss(b):
	def user_losses(y_true, y_pred):
         loss_au=categorical_crossentropy(y_true, y_pred)
         loss_abu=abs(loss_au)          
         user_ls=b*K.sum(loss_abu)
         return user_ls
	return user_losses
 

def group_loss(b):
	def group_losses(y_true, y_pred):
         loss_ag=categorical_crossentropy(y_true, y_pred)
         loss_abg=abs(loss_ag)          
         group_ls=b*K.sum(loss_abg)
         return group_ls
	return group_losses