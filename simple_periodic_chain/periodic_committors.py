#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:41:30 2019

@author: bzfhelfm
"""
import numpy as np
import scipy as sp

#A: left well <=-0.5
#B: right well >=0.5
#C: transition region (-0.5, 0.5)
A = [-2,-0.5] #1
B = [0.5,2] #2
#C = [-0.5, 0.5] #0


def set_A(x,A):
    if x<=A[1]:
        return 1
    else:
        return 0
    
def set_B(x,B):
    if x>=B[0]:
        return 1
    else:
        return 0
    
def set_C(x,A,B):
    if set_A(x,A) ==0 and set_B(x,B) ==0:
        return 1
    else:
        return 0
    
#set_member = np.array([set_A(x,A) for x in xs]) +  2*np.array([set_B(x,B) for x in xs])


def periodic_committors(C, A,B,xs):
    dim=np.shape(C)[0] #dimension of the state space

    period= np.shape(C)[2] 
    
    #indices of transition region C
    ind_C = np.array([set_C(x,A,B) for x in xs])
    dim_C = np.sum(ind_C)
    #indices of B
    ind_B = np.array([set_B(x,B) for x in xs])
    dim_B = np.sum(ind_B)
    
    #assemble ps?C into PA_C , first on diagonal than shift!
    PA_C_diag = np.zeros((period*dim_C, period*dim_C))
    PA_C = np.zeros((period*dim_C, period*dim_C))
    
        
    #assemble b
    b=np.zeros(dim_C*period)
    
    
    for p in range(period):
        #cut submatrices with indices i,j \in C
        P=C[:,:,p]
        if np.sum(P[:,0])!=1.:
            P = np.transpose(P) #should be columnsum
        P1 = P[ind_C==1,:]
        P_C = P1[:,ind_C==1]
        PA_C_diag[p*dim_C:(p+1)*dim_C,p*dim_C:(p+1)*dim_C]=P_C
        
        P2= P[ind_B==1,:]
        P_B = P2[:,ind_C==1]
        #assemble b assuming that the P_Bs are column sum, the P_Bs give the probabilites to transition from i in C to j in B
        b[p*dim_C:(p+1)*dim_C] = np.sum(P_B,0)
    
    PA_C[0:dim_C,:]=PA_C_diag[(period-1)*dim_C:(period)*dim_C,:] 
    PA_C[dim_C:period*dim_C,:] = PA_C_diag[0:(period-1)*dim_C,:]
    
    A= np.diag(np.ones(dim_C*period)) - PA_C
    
    #invert A
    inv_A=np.linalg.inv(A)
    #find q
    qT=b.dot(inv_A)
    
    qs=np.zeros((dim,period))
    for p in range(period):
        qs[ind_B==1,p]=1
        qs[ind_C==1,p]=qT[p*dim_C:(p+1)*dim_C]

    return qs