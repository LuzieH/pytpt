#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:41:30 2019

@author: bzfhelfm
"""
import numpy as np



def periodic_committors_stacked(T,period_assign, ind_B, ind_C):
    dim=np.shape(T)[0] #dimension of the state space
    dim_T= np.shape(T)[2] 
    period=len(period_assign)
    dim_C = np.sum(ind_C)    
    dim_B = np.sum(ind_B)
    
    #entries for jumps from one period to the next
    P_mult_C=np.diag(np.ones(dim_C)) # multiplied matrix over all times with only transitions in C
    P_mult_BC=np.zeros((dim_B, dim_C,period)) # multiplied matrix until time m, with prob from C to B at last time   
    
    #all intermediate matrices
    P_C_all=np.zeros((dim_C,dim_C, dim_T)) #single transition matrices restr. to C
    P_BC_all=np.zeros((dim_B, dim_C, dim_T)) # single trans. matr. restric. to jumps from C to B
    
    #constructing restricted matrices for each dim_T
    for d in range(dim_T):
        P=T[:,:,d]
        if np.sum(P[:,0])!=1.:
            P = np.transpose(P) #should be columnsum
       
        #cut submatrices with indices i,j \in C
        P1 = P[ind_C==1,:]
        P_C = P1[:,ind_C==1]
        P_C_all[:,:,d]=P_C
        
        #cut submatrices with indices i in C, j in B
        P2= P[ind_B==1,:]
        P_B = P2[:,ind_C==1]        
        P_BC_all[:,:,d]=P_B
       
    for p in range(period):
 
        P_B=P_BC_all[:,:,int(period_assign[p])]
        P_C=P_C_all[:,:,int(period_assign[p])]
        
        P_mult_BC[:,:,p]=P_B.dot(P_mult_C)
        P_mult_C=P_C.dot(P_mult_C)
 
    A = np.diag(np.ones(dim_C)) - P_mult_C
    
    #invert A
    inv_A = np.linalg.inv(A)
    #find q_0
    
    #define remaining part of equation as b
    b= np.zeros(dim_C)
    for p in range(period):
        b=b+np.ones(dim_B).dot(P_mult_BC[:,:,p])
        

    qs_C=np.zeros((dim_C, period+1))
    qs_C[:,0]=b.dot(inv_A)
    qs_C[:,period]=qs_C[:,0]  
    
    for p in range(period-1):
        qs_C[:,period-p-1] =qs_C[:,period-p].dot(P_C_all[:,:,int(period_assign[period-p-1])]) + np.ones(dim_B).dot(P_BC_all[:,:,int(period_assign[period-p-1])])
    
    qs=np.zeros((dim,period))
    for p in range(period):
        qs[ind_B==1,p]=1
        qs[ind_C==1,p]=qs_C[:,p]
        
    return qs,A




#def periodic_committors(T, ind_B, ind_C):
#    dim=np.shape(T)[0] #dimension of the state space
#
#    period= np.shape(T)[2] 
#    
#    dim_C = np.sum(ind_C)    
#    #dim_B = np.sum(ind_B)
#    
#    #assemble ps?C into PA_C , first on diagonal than shift!
#    PA_C_diag = np.zeros((period*dim_C, period*dim_C))
#    PA_C = np.zeros((period*dim_C, period*dim_C))
#    
#        
#    #assemble b
#    b=np.zeros(dim_C*period)
#    
#    
#    for p in range(period):
#        #cut submatrices with indices i,j \in C
#        P=T[:,:,p]
#        if np.sum(P[:,0])!=1.:
#            P = np.transpose(P) #should be columnsum
#        P1 = P[ind_C==1,:]
#        P_C = P1[:,ind_C==1]
#        PA_C_diag[p*dim_C:(p+1)*dim_C,p*dim_C:(p+1)*dim_C]=P_C
#        
#        P2= P[ind_B==1,:]
#        P_B = P2[:,ind_C==1]
#        #assemble b assuming that the P_Bs are column sum, the P_Bs give the probabilites to transition from i in C to j in B
#        b[p*dim_C:(p+1)*dim_C] = np.sum(P_B,0)
#    
#    PA_C[0:dim_C,:]=PA_C_diag[(period-1)*dim_C:(period)*dim_C,:] 
#    PA_C[dim_C:period*dim_C,:] = PA_C_diag[0:(period-1)*dim_C,:]
#    
#    A= np.diag(np.ones(dim_C*period)) - PA_C
#    
#    #invert A
#    inv_A=np.linalg.inv(A)
#    #find q
#    qT=b.dot(inv_A)
#    
#    qs=np.zeros((dim,period))
#    for p in range(period):
#        qs[ind_B==1,p]=1
#        qs[ind_C==1,p]=qT[p*dim_C:(p+1)*dim_C]
#
#    return qs,PA_C

