#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:24:53 2019

@author: bzfhelfm
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix

##############################################################################
#triple well in 3D

factor=0.25
V_param = lambda x,y,p: factor*(3*np.exp(-x**2-(y-(1./3))**2) - p*np.exp(-x**2-(y-(5./3))**2) - 5*np.exp(-(x-1)**2-y**2) - 5*np.exp(-(x+1)**2-y**2)  + 0.2*(x**4) + 0.2*(y-1./3)**4)

V0 = lambda x,y: V_param(x,y,3)
V1 = lambda x,y: V_param(x,y,3.5)
V2 = lambda x,y: V_param(x,y,4)
V3 = lambda x,y: V_param(x,y,4.5)

dV_param_x = lambda x,y,p: factor*((-2*3*x)*np.exp(-x**2-(y-(1./3))**2) +(p*2*x)*np.exp(-x**2-(y-(5./3))**2) + (10*(x-1))*np.exp(-(x-1)**2-y**2) + (10*(x+1))*np.exp(-(x+1)**2-y**2)  + 0.8*(x**3))
dV_param_y = lambda x,y,p: factor*((-2*3*(y-1./3))*np.exp(-x**2-(y-(1./3))**2) + (p*2*(y-(5./3)))*np.exp(-x**2-(y-(5./3))**2) + (10*y)*np.exp(-(x-1)**2-y**2) + (10*y)*np.exp(-(x+1)**2-y**2)  + 0.8*(y-1./3)**3)

dV0 = lambda x,y: np.array([dV_param_x(x,y,3),dV_param_y(x,y,3)])

dV1 = lambda x,y: np.array([dV_param_x(x,y,4),dV_param_y(x,y,3.5)])

dV2 = lambda x,y: np.array([dV_param_x(x,y,4.5),dV_param_y(x,y,4)])

dV3 = lambda x,y: np.array([dV_param_x(x,y,5),dV_param_y(x,y,4.5)])

############################################################################
#transition matrices

dVs =[dV0, dV1, dV2, dV3]#, dV2, dV1]


period = 4
interval = np.array([[-2,2],[-2,3]])
dim=np.shape(interval)[0]
dx_power=1
dx=1./(10**dx_power)

x = np.arange(-2,2+dx, dx)
y = np.arange(-2,3+dx, dx)
xv, yv = np.meshgrid(x, y)

xdim = np.shape(xv)[0]
ydim = np.shape(xv)[1]


xn=np.reshape(xv,(xdim*ydim,1))
yn=np.reshape(yv,(xdim*ydim,1))

grid = np.squeeze(np.array([xn,yn]))

Nstep=50
sigma=1 #0.1
dt=0.01
lag=10
dim_st = xdim*ydim #dimension of the statespace

C=transitionmatrix_2D(dVs,period, sigma, dt, lag, Nstep, interval, dx_power,x,y,dim)
P0=np.transpose(C[:,:,0]) #columnsum
P1=np.transpose(C[:,:,1])
P2=np.transpose(C[:,:,2])
P3=np.transpose(C[:,:,3])
#P4=np.transpose(C[:,:,4])
#P5=np.transpose(C[:,:,5])

################################################################################
#stationary densiites
P_mult = P1.dot(P2.dot(P3.dot(P2.dot(P1.dot(P0)))))
x_0=1./(dim_st*dx*dx)*np.ones(dim_st) # uniform starting distribution
Pk=np.linalg.matrix_power(P_mult,10)
pi0=Pk.dot(x_0) #"periodic" stationary distributions
pi1=P1.dot(pi0)
pi2=P2.dot(pi1)
pi3=P3.dot(pi2)
pi4=P2.dot(pi3)
pi5=P3.dot(pi4)

fig2 = plt.figure()
plt.imshow(np.reshape(pi0,(xdim,ydim)))
plt.show()


#V0_dx=np.reshape(V0(xn,yn),(xdim,ydim))
##V3_dx=np.reshape(V3(xn,yn),(xdim,ydim))
#
#fig1 = plt.figure()
#plt.imshow(V0_dx)
#plt.show()
##
#
#fig2 = plt.figure()
#plt.imshow(V3_dx)
#plt.show()

#plt.scatter(x,dV(x))

################################################################################
#committors


#define by center and radius!
A_center = np.array([-1,0])  
B_center = np.array([1,0])
radius_setAB = 0.4

def set_A_triplewell(x,A_center, radius_setAB):
    if np.linalg.norm(x-A_center) <=radius_setAB:
        return 1
    else:
        return 0
    
def set_B_triplewell(x,B_center, radius_setAB):
    if np.linalg.norm(x-B_center)<=radius_setAB:
        return 1
    else:
        return 0
    
def set_C_triplewell(x,A_center, B_center, radius_setAB):
    if set_A_triplewell(x,A_center, radius_setAB) ==0 and set_B_triplewell(x,B_center, radius_setAB) ==0:
        return 1
    else:
        return 0

#indices of transition region C
ind_C = np.array([set_C_triplewell(grid[:,i],A_center,B_center,radius_setAB) for i in np.arange(np.shape(xn)[0])])

#indices of B
ind_B = np.array([set_B_triplewell(grid[:,i],B_center, radius_setAB) for i in np.arange(np.shape(xn)[0])])

period2 = 6
C2= np.zeros((np.shape(C)[0], np.shape(C)[1],period2))
C2[:,:,0:4]=C
C2[:,:,4]=C[:,:,2]
C2[:,:,5]=C[:,:,1]

[qs,P_A]=periodic_committors_stacked(C2,ind_B, ind_C)


fig1 = plt.figure()
plt.imshow(np.reshape(qs[:,0],(xdim,ydim)))
plt.show()

fig2 = plt.figure()
plt.imshow(np.reshape(qs[:,1],(xdim,ydim)))
plt.show()

fig3 = plt.figure()
plt.imshow(np.reshape(qs[:,2],(xdim,ydim)))
plt.show()

fig4 = plt.figure()
plt.imshow(np.reshape(qs[:,3],(xdim,ydim)))
plt.show()

###############################################################################
#reactive probability

m = np.multiply(qs[:,0],np.multiply(pi0,1-qs[:,0]))
plt.imshow(np.reshape(m,(xdim,ydim)))



##############################################################################
#probability current