#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:30:43 2019

@author: bzfhelfm
"""

import numpy as np
#simple periodic markov chain with period 2 and three states

####################################################
#stationary density for periodic chain


#augmented transition matrix
#P1=np.array([[0.8, 0.5, 0],[0.2, 0, 0.2],[0, 0.5, 0.8]])
#P2=np.array([[0.9, 0.5, 0],[0.1, 0, 0.3],[0, 0.5, 0.7]])
#PA=np.zeros((6,6))
#PA[3:6,0:3]=P1
#PA[0:3,3:6]=P2 #already column normalized

#and with 4 states
P1=np.array([[0.8, 0.5, 0.4, 0],[0.15, 0,0.1, 0.05],[0.05, 0.1,0, 0.15],[0, 0.4, 0.5, 0.8]])
P2=np.array([[0.9, 0.5, 0.3, 0],[0.05, 0,0.2, 0.15],[0.05, 0.1,0, 0.15],[0, 0.4, 0.5, 0.7]])
PA=np.zeros((8,8))
PA[4:8,0:4]=P1
PA[0:4,4:8]=P2 #already column normalized


####################################################
#committor functions

# state A is 1, transition region is 2 and state B is 3
q1=np.array([0,0.4,0.6,1])
q2=np.array([0,0.4,0.6,1])
Q=np.zeros(8)
Q[0:4]=q1
Q[4:8]=q2

for i in range(1000):
    for j in np.array([1,2]):
        Q[j]=P1[:,j].dot(Q[4:8])
        
    for k in np.array([5,6]):
        Q[k]=P2[:,k-4].dot(Q[0:4])    
        
print(Q)

#Q=np.zeros(8)
#Q[0:4]=q1
#Q[4:8]=q2
#
##only committors on transition region are affected by changes
#Qchanges=np.array([1,1])
#PQ=np.zeros((8,8))
#PQ[0:4,4:8]=P1
#PQ[4:8,0:4]=P2
#
#PQchanges=np.zeros((4,8))
#PQchanges[0,:]=PQ[1,:]
#PQchanges[1,:]=PQ[2,:]
#PQchanges[2,:]=PQ[5,:]
#PQchanges[3,:]=PQ[6,:]
#
#for i in range(1000):
#    Qchanges=PQchanges.dot(Q)    
#    Q[1]=Qchanges[0]
#    Q[2]=Qchanges[1]
#    Q[5]=Qchanges[2]
#    Q[6]=Qchanges[3]
#    
#print(Q)  