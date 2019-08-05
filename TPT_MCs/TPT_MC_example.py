#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:25:01 2019

@author: bzfhelfm
"""
import numpy as np
import matplotlib.pyplot as plt
import TPT

#load example data
P=np.load('/nfs/numerik/bzfhelfm/Dokumente/PhD/code/transitions/TPT_MCs/P.npy')
ind_A=np.load('/nfs/numerik/bzfhelfm/Dokumente/PhD/code/transitions/TPT_MCs/ind_A.npy')
ind_B=np.load('/nfs/numerik/bzfhelfm/Dokumente/PhD/code/transitions/TPT_MCs/ind_B.npy')
ind_C=np.load('/nfs/numerik/bzfhelfm/Dokumente/PhD/code/transitions/TPT_MCs/ind_C.npy')
S_shape=np.load('/nfs/numerik/bzfhelfm/Dokumente/PhD/code/transitions/TPT_MCs/S_shape.npy')

#instanciate
triplewell=TPT.TPT_MC(P,ind_A, ind_B, ind_C)

#compute committor probabilities
[q_f,q_b]=triplewell.committor()

#stationary density
stat_dens=triplewell.stat_dens

fig1 = plt.figure()
plt.imshow(np.reshape(stat_dens,(S_shape)))

fig2 = plt.figure()
plt.imshow(np.reshape(q_f,(S_shape)))

fig3 = plt.figure()
plt.imshow(np.reshape(q_b,(S_shape)))

#reactive density
reac_dens=triplewell.reac_density()
#currents
[current,eff_current]=triplewell.reac_current()
current_dens=triplewell.current_density()
#rate
rate=triplewell.transition_rate()

fig4 = plt.figure()
plt.imshow(np.reshape(reac_dens,(S_shape)))

fig5 = plt.figure()
plt.imshow(np.reshape(current_dens,(S_shape)))


#rate matrix
#tau=0.1
#R=1./tau *(P-np.diag(np.ones(S)))
#
#[current2,eff_current2]=reac_current(q_f,q_b,stat_dens,R)
#current2[np.arange(S),np.arange(S)]=0
#current_dens2=current_density(np.abs(current2-np.transpose(current2)))
#plt.imshow(np.reshape(current_dens2,(51,41)))