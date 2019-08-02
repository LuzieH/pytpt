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
V1 = lambda x,y: V_param(x,y,2.5)
V2 = lambda x,y: V_param(x,y,1.75)
V3 = lambda x,y: V_param(x,y,1)

dV_param_x = lambda x,y,p: factor*((-2*3*x)*np.exp(-x**2-(y-(1./3))**2) +(p*2*x)*np.exp(-x**2-(y-(5./3))**2) + (10*(x-1))*np.exp(-(x-1)**2-y**2) + (10*(x+1))*np.exp(-(x+1)**2-y**2)  + 0.8*(x**3))
dV_param_y = lambda x,y,p: factor*((-2*3*(y-1./3))*np.exp(-x**2-(y-(1./3))**2) + (p*2*(y-(5./3)))*np.exp(-x**2-(y-(5./3))**2) + (10*y)*np.exp(-(x-1)**2-y**2) + (10*y)*np.exp(-(x+1)**2-y**2)  + 0.8*(y-1./3)**3)

dV0 = lambda x,y: np.array([dV_param_x(x,y,3),dV_param_y(x,y,3)])

dV1 = lambda x,y: np.array([dV_param_x(x,y,4),dV_param_y(x,y,3.5)])

dV2 = lambda x,y: np.array([dV_param_x(x,y,4.5),dV_param_y(x,y,4)])

dV3 = lambda x,y: np.array([dV_param_x(x,y,5),dV_param_y(x,y,4.5)])


############################################################################
#transition matrices


dVs =[dV0, dV1, dV2, dV3]#forces
dim_dV = 4
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

Nstep=100
sigma=1 #0.1
dt=0.01
lag=10
dim_st = xdim*ydim #dimension of the statespace

#transition matrices
C=transitionmatrix_2D(dVs,dim_dV, sigma, dt, lag, Nstep, interval, dx_power,x,y,dim)
P0=csc_matrix(np.transpose(C[:,:,0])) #columnsum
P1=csc_matrix(np.transpose(C[:,:,1]))
P2=csc_matrix(np.transpose(C[:,:,2]))
P3=csc_matrix(np.transpose(C[:,:,3]))
Ps_sparse = [P0, P1, P2, P3]

#################################################################################
#plotting potentials

V0_plot=V0(xn,yn)
V3_plot=V3(xn,yn)
min_value=min(np.min(V0_plot), np.min(V3_plot))
max_value= max(np.max(V0_plot), np.max(V3_plot))

fig2 = plt.figure()
plt.imshow(np.reshape(V0_plot,(xdim,ydim)),vmin=min_value, vmax=max_value)
plt.savefig('V0.png')
plt.show()

fig3 = plt.figure()
plt.imshow(np.reshape(V3_plot,(xdim,ydim)), vmin=min_value, vmax=max_value)
plt.savefig('V3.png')
plt.show()

################################################################################
#stationary densiites
rep=15 #each transition matrix is applied rep times during the period
rep2=10

#assign the matrices to time points over the period
period=rep*3 + rep2*3
period_assign=np.zeros(period)

period_assign[0:rep]=0
period_assign[rep:2*rep]=1
period_assign[2*rep:2*rep+rep2]=2
period_assign[2*rep +rep2:2*rep +2*rep2]=3
period_assign[2*rep +2*rep2:2*rep +3*rep2]=2
period_assign[2*rep +3*rep2:3*rep +3*rep2]=1
 

pis=np.zeros((dim_st,period)) #periodic stationary densities
#transition matrix over one period
P_mult=csc_matrix((np.ones(dim_st),(np.arange(dim_st),np.arange(dim_st))))
for p in range(period):
    P_mult=(Ps_sparse[int(period_assign[p])]).dot(P_mult)
    
x_0=1./(dim_st*dx*dx)*np.ones(dim_st) # uniform starting distribution

Pk=P_mult**10#np.linalg.matrix_power(P_mult,10)

#"periodic" stationary distributions
pi0=Pk.dot(x_0) 
pis[:,0]=pi0
for p in range(1,period):
    pis[:,p]=(Ps_sparse[int(period_assign[p])]).dot(pis[:,p-1])

#plotting stat dens
min_value=np.min(pis)
max_value= np.max(pis)

fig2 = plt.figure()
plt.imshow(np.reshape(pis[:,0],(xdim,ydim)),vmin=min_value, vmax=max_value)
plt.savefig('stat_dens_0.png')
plt.show()

fig3 = plt.figure()
plt.imshow(np.reshape(pis[:,45],(xdim,ydim)), vmin=min_value, vmax=max_value)
plt.savefig('stat_dens_45.png')
plt.show()

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
 
#committors
[qs,A]=periodic_committors_stacked(C,period_assign,ind_B, ind_C)

###############################################################################
#plotting committors
fig1 = plt.figure()
plt.imshow(np.reshape(qs[:,0],(xdim,ydim)),vmin=0, vmax=1)
plt.savefig('comm_0.png')
plt.show()


fig4 = plt.figure()
plt.imshow(np.reshape(qs[:,45],(xdim,ydim)),vmin=0, vmax=1)
plt.savefig('comm_45.png')
plt.show()
 
 
################################################################################
##reactive probability
#
#m = np.multiply(qs[:,0],np.multiply(pi0,1-qs[:,0]))
#plt.imshow(np.reshape(m,(xdim,ydim)))
#
##its an approximation, also need to compute backward committors!!!
#qs_min=1-qs
#
#reac_pis= np.zeros((dim_st,period))
#for p in range(period):
#    reac_pis[:,p]= np.multiply(qs[:,p],np.multiply(pis[:,p],qs_min[:,p])
#    
###############################################################################
##probability current
#current = np.zeros((dim_st, dim_st, period))
#for p in range(period-1):
#    for i in range(dim_st):
#        for j in range(dim_st):
#            current[i,j,p]=qs_min[i,p]*pis[i,p]*Ps_sparse[int(period_assign[p])][i,j]*qs[i,p+1]
#
#for i in range(dim_st):
#    for j in range(dim_st):
#        current[i,j,period-1]=qs_min[i,period-1]*pis[i,period-1]*Ps_sparse[int(period_assign[period-1])][i,j]*qs[i,0]    