#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:02:41 2019

@author: bzfhelfm
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
######################################################################
#periodic double well potential in 1D on [-2, 2] where parameter p varies
factor=2
V_param = lambda x,p: factor*(x**2-1)**2+p*x
dV_param = lambda x,p: factor* 4*x*(x**2-1)+ p

V0 = lambda x,y: V_param(x,y,0)
V1 = lambda x,y: V_param(x,y,0.25)
V2 = lambda x,y: V_param(x,y,0.5)
V3 = lambda x,y: V_param(x,y,0.7)

dV0 = lambda x,y: dV_param(x,y,0)
dV1 = lambda x,y: dV_param(x,y,0.25)
dV2 = lambda x,y: dV_param(x,y,0.5)
dV3 = lambda x,y: dV_param(x,y,0.7)

######################################################################
#TRANSITION MATRICES

dVs =[dV0, dV1, dV2, dV3] 
diff_dV = 4

interval = [-2,2]
dx_power=2
dx=1./(10**dx_power) #space discretization
xs = np.round(np.arange(interval[0], interval[1]+dx, dx),dx_power)
dim=np.shape(xs)[0]
Nstep=10000 #number of seeds per grid cell (for constructing P)
sigma=0.8 #0.1
dt=0.01
lag=10 #time between steps later will be lag*dt

C = transitionmatrix_1D(dVs,diff_dV, sigma, dt, lag,Nstep, interval, dx_power)

 
###################################################################### 
#COMMITTORS


#A: left well <=-0.5
#B: right well >=0.5
#C: transition region (-0.5, 0.5)
A = [-2,-0.5] #1
B = [0.5,2] #2
#C = [-0.5, 0.5] #0


def set_A_1D(x,A):
    if x<=A[1]:
        return 1
    else:
        return 0
    
def set_B_1D(x,B):
    if x>=B[0]:
        return 1
    else:
        return 0
    
def set_C_1D(x,A,B):
    if set_A_1D(x,A) ==0 and set_B_1D(x,B) ==0:
        return 1
    else:
        return 0

#indices of transition region C
ind_C = np.array([set_C_1D(x,A,B) for x in xs])
#indices of B
ind_B = np.array([set_B_1D(x,B) for x in xs])


#the transition matrices might be repeated several times, rep/rep2 times
rep=1
rep2=1 
period2=rep*3 + rep2*3

#period_assign assignes each time point during period the 
#corresponding transtition matrix
period_assign=np.zeros(period2)

period_assign[0:rep]=0
period_assign[rep:2*rep]=1
period_assign[2*rep:2*rep+rep2]=2
period_assign[2*rep +rep2:2*rep +2*rep2]=3
period_assign[2*rep +2*rep2:2*rep +3*rep2]=2
period_assign[2*rep +3*rep2:3*rep +3*rep2]=1
 

[qs,P_A]=periodic_committors_stacked(C,period_assign,ind_B, ind_C)



##########################################################################
#PLOTTING

fig1 = plt.figure()
plt.plot(xs,V0(xs),label='V_0')
plt.plot(xs,V1(xs),label='V_1=V_5')
plt.plot(xs,V2(xs),label='V_2=V_4')
plt.plot(xs,V3(xs),label='V_3')
plt.legend()
plt.title('Potentials')
plt.show()

fig2 = plt.figure()
plt.plot(xs,qs[:,0],label='q_0')
plt.plot(xs,qs[:,1],label='q_1')
plt.plot(xs,qs[:,2],label='q_2')
plt.plot(xs,qs[:,3],label='q_3')
plt.plot(xs,qs[:,4],label='q_4')
plt.plot(xs,qs[:,5],label='q_5')
plt.legend()
plt.title('Committors')
plt.show()

###########################################################################
#STATIONARY DENSITIES
P0=np.transpose(C[:,:,0]) #columnsum
P1=np.transpose(C[:,:,1])
P2=np.transpose(C[:,:,2])
P3=np.transpose(C[:,:,3])
P4=P2 #np.transpose(C[:,:,4])
P5=P1#np.transpose(C[:,:,5])


P_mult = P5.dot(P4.dot(P3.dot(P2.dot(P1.dot(P0))))) #transition matrix over one period
x_0=1./(dim*dx)*np.ones(dim) # uniform starting distribution
Pk=np.linalg.matrix_power(P_mult,10000)
#"periodic" stationary distributions
pi0=Pk.dot(x_0) 
pi1=P1.dot(pi0)
pi2=P2.dot(pi1)
pi3=P3.dot(pi2)
pi4=P4.dot(pi3)
pi5=P5.dot(pi4)

fig3 = plt.figure()
plt.plot(xs,pi0,label='pi_0')
plt.plot(xs,pi1,label='pi_1')
plt.plot(xs,pi2,label='pi_2')
plt.plot(xs,pi3,label='pi_3')
plt.plot(xs,pi4,label='pi_4')
plt.plot(xs,pi5,label='pi_5')
plt.legend()
plt.title('Periodic Stationary Densities')
plt.show()
