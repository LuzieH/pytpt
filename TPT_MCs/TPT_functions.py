#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:56:21 2019

@author: bzfhelfm
"""

import numpy as np

#transition matirx needs to irreducible and normalized (rowsum)
#given transition matrix (rowsum?), indices of A and B, C
#S=6 #size of the statespace

#P=np.random.rand(S,S)
#P=P/np.sum(P,axis=1).reshape(S,1)


def committor(P,ind_A, ind_B, ind_C):
    #ind_A etc are lists with 1,0 entries depending on whether the state belongs to the set or not
    #check if transition matrix is normalized and rowsum
    
    #state space size
    S=np.shape(P)[0]
    #compute stationary density
    eigv,eig = np.linalg.eig(np.transpose(P))
    stat_dens=np.real(eig[:,0])/np.sum(np.real(eig[:,0]))
    
    #compute backward transition matrix    
    P_back=np.zeros(np.shape(P))
    for i in np.arange(S):
        for j in np.arange(S):
             P_back[j,i]=P[i,j]*stat_dens[i]/stat_dens[j]
    
    #transition matrices from states in C to C
    P_C=P[np.ix_(ind_C,ind_C)]
    P_back_C=P_back[np.ix_(ind_C,ind_C)]
    
    #amd from C to B
    P_CB=P[np.ix_(ind_C,ind_B)]
    P_back_CA=P_back[np.ix_(ind_C,ind_A)]
    
    #forward committor on C, the transition region
    q_f_C=np.zeros(int(np.size(ind_C)))
    b=np.sum(P_CB,axis=1)
    inv1=np.linalg.inv(np.diag(np.ones(np.size(ind_C)))-P_C)
    q_f_C=inv1.dot(b)
    #add entries to forward committor vector (which is 0 on A, 1 on B)
    q_f = np.zeros(S)
    q_f[np.ix_(ind_B)]=1 
    q_f[np.ix_(ind_C)] =q_f_C
    
    #backward committor    
    q_b_C=np.zeros(int(np.size(ind_C)))
    a=np.sum(P_back_CA,axis=1)
    inv2=np.linalg.inv(np.diag(np.ones(np.size(ind_C)))-P_back_C)
    q_b_C=inv2.dot(a)
    
    q_b = np.zeros(S)
    q_b[np.ix_(ind_A)]=1 
    q_b[np.ix_(ind_C)] =q_b_C
    
    return stat_dens, P_back, q_f, q_b 


#give example by importing some transition matrix, plot committors, reactive density, effective flux per cell

#new function that computes statistics
    #normalized reactive density
def reac_density(q_f,q_b,stat_dens):
    reac_dens=np.multiply(q_b,np.multiply(stat_dens,q_f))
    reac_dens=reac_dens/np.sum(reac_dens) #normalization
    return reac_dens

def reac_current(q_f,q_b,stat_dens,P):
    S=np.shape(P)[0]
    current=np.zeros(np.shape(P))
    eff_current=np.zeros(np.shape(P))
    for i in np.arange(S):
        for j in np.arange(S):
            current[i,j]=stat_dens[i]*q_b[i]*P[i,j]*q_f[j]
            if i+1>j:
                eff_current[i,j]=np.max([0,current[i,j]-current[j,i]])
                eff_current[j,i]=np.max([0,current[j,i]-current[i,j]])
    return current, eff_current

def transition_rate(current, ind_A):
    rate= np.sum(current[ind_A,:])
    return rate

def current_density(eff_current):
    S=np.shape(eff_current)[0]
    current_dens=np.zeros(S)
    for i in ind_C:
        current_dens[i]=np.sum(eff_current[i,:])
    return current_dens