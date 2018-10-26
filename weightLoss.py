#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 20:55:25 2018

@author: liuhuihui
"""

from keras import backend as K
from keras.objectives import categorical_crossentropy

def tag_loss(a):
	def tag_losses(y_true, y_pred):
         loss_a=categorical_crossentropy(y_true, y_pred)         
         return a*loss_a
	return tag_losses


def user_loss(b):
	def user_losses(y_true, y_pred):
         loss_au=categorical_crossentropy(y_true, y_pred)
         return b*loss_au
	return user_losses
 

def group_loss(c):
	def group_losses(y_true, y_pred):
         loss_ag=categorical_crossentropy(y_true, y_pred)
         return c*loss_ag
	return group_losses