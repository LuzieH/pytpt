#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:00:49 2019

@author: bzfhelfm
"""
import numpy as np
import scipy as sp
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

#what is the long term behaviour if we start the chain in [0.5 0 0.5]
start=np.array([0.5,0,0,0.5])

#stationary density for P1
end1=start
for i in range(1000):
    end1=P1.dot(end1)
print(end1)
    
#stationary density for P2
end2=start
for i in range(1000):
    end2=P2.dot(end2)  
print(end2)    

#using augmented matrix
startA=np.array([0.1,0,0,0.5,0.15,0,0,0.25])
#initial density chosen st equal mass at both times
endA=startA
for i in range(1000):
    endA=PA.dot(endA)  
print(endA*2)    

# does this agree with the dominant right eigenvecotr?
eigv,eig = np.linalg.eig(PA)
#PA.dot(eig[:,0])==eig[:,0]
eig[:,0]/endA #--> are multiplicatives of each other


#irreducible (every state reachable form every other state) -> stationary density exists
#but periodic, some states are only reachible in even times -> not every initial density converges to stationary density 
#-> depends on where the mass lies
#but how about the not augmented systems? does then every initial density converge?
