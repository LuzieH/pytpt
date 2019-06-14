#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:02:41 2019

@author: bzfhelfm
"""

import numpy as np
import matplotlib.pyplot as plt
#periodic double well potential in 1D on [-2, 2]
factor=1.5
#V1 = lambda x: factor*(x**2-1)**2
#dV1 = lambda x: factor*4*x*(x*2-1)
#
#V2 = lambda x: factor*(x**2-1)**2 +0.1*x
#dV2 = lambda x: factor*4*x*(x*2-1) + 0.1
#
#V3 = lambda x: factor*(x**2-1)**2 + 0.2*x
#dV3 = lambda x: factor*4*x*(x*2-1) + 0.2
#
#
#dVs = [V1, V2, V3, V2]
period = 4
dVs=list()
#dVs=[0]*period
for p in range(period):
    Vnew= lambda x: factor*4*x*(x*2-1) +  0.1* np.sin(p*np.pi/period)
    dVs.append(Vnew)
#
#x = np.arange(-2,2,0.1)
#plt.scatter(x,V1(x))
#plt.scatter(x,V2(x))
#plt.scatter(x,V3(x))
#plt.scatter(x,dV(x))


interval = [-1.5, 1.5]
dx_power=2
dx=1./(10**dx_power)
Nstep=1000
sigma=1.2
dt=1

C = transitionmatrix_1D(dVs,period, sigma, dt, Nstep, interval, dx_power)
gridsize = np.shape(C)[0]
C_augmented = np.zeros((gridsize*period,gridsize*period))
C_augmented[0:gridsize,gridsize:2*gridsize]= C[:,:,0]
C_augmented[gridsize:2*gridsize,2*gridsize:3*gridsize]= C[:,:,1]
C_augmented[2*gridsize:3*gridsize,3*gridsize:4*gridsize]= C[:,:,2]
C_augmented[3*gridsize:4*gridsize,0:gridsize]= C[:,:,3]
C_augmented=(1./Nstep)*C_augmented
#todo make sparse
#use voronoi cells, irrgeular, st all row sums are >0
#start with uniform seeding several times and count the transitions
#plt.figure(1)
#traj_X=np.reshape(traj,(period*Nstep))
#plt.figure(2)
#plt.scatter(range(period*Nstep),traj_X)