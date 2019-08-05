#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:56:21 2019

@author: bzfhelfm
"""

import numpy as np



class TPT_MC:
    """Calculates committor probabilities and transition statistics of 
    Markov chain models"""

    def __init__(self, P, ind_A, ind_B,  ind_C):
        """
        Initialize an instance by defining the transition matrix and the sets 
        between which the transition statistics should be computed.
        
        Parameters:
        P: array
            irreducible and row-stochastic (rows sum to 1) transition matrix  
            of size S x S, S is the size of the state space St={1,2,...,S} 
        ind_A: array
            set of indices of the state space that belong to the set A
        ind_B: array
            set of indices of the state space that belong to the set B
        ind_C: array
            set of indices of the state space that belong to the transition 
            region C, i.e. the set C = St\(A u B)
        """
 
        self.P=P
        self.ind_A=ind_A
        self.ind_B=ind_B
        self.ind_C=ind_C
        self.S=np.shape(self.P)[0]
        #self._P_back=None

        
    def committor(self):
        """
        Function that computes the forward committor q_f (probability that the 
        particle will next go to B rather than A) and backward commitor q_b 
        (probability that the system last came from A rather than B).
        """
         
        #compute stationary density
        eigv,eig = np.linalg.eig(np.transpose(self.P))
        self.stat_dens=np.real(eig[:,0])/np.sum(np.real(eig[:,0]))
        
        #compute backward transition matrix    
        P_back=np.zeros(np.shape(self.P))
        for i in np.arange(self.S):
            for j in np.arange(self.S):
                 P_back[j,i]=self.P[i,j]*self.stat_dens[i]/self.stat_dens[j]
        self.P_back=P_back
        
        #transition matrices from states in C to C
        P_C=self.P[np.ix_(self.ind_C,self.ind_C)]
        P_back_C=self.P_back[np.ix_(self.ind_C,self.ind_C)]
        
        #amd from C to B
        P_CB=self.P[np.ix_(self.ind_C,self.ind_B)]
        P_back_CA=P_back[np.ix_(self.ind_C,self.ind_A)]
        
        #forward committor on C, the transition region
        q_f_C=np.zeros(int(np.size(self.ind_C)))
        b=np.sum(P_CB,axis=1)
        inv1=np.linalg.inv(np.diag(np.ones(np.size(self.ind_C)))-P_C)
        q_f_C=inv1.dot(b)
        
        #add entries to forward committor vector on A, B (which is 0 on A, 1 on B)
        q_f = np.zeros(self.S)
        q_f[np.ix_(self.ind_B)]=1 
        q_f[np.ix_(self.ind_C)] =q_f_C
        
        #backward committor    
        q_b_C=np.zeros(int(np.size(self.ind_C)))
        a=np.sum(P_back_CA,axis=1)
        inv2=np.linalg.inv(np.diag(np.ones(np.size(self.ind_C)))-P_back_C)
        q_b_C=inv2.dot(a)
        
        #add entries to forward committor vector on A, B (which is 1 on A, 0 on B)
        q_b = np.zeros(self.S)
        q_b[np.ix_(self.ind_A)]=1 
        q_b[np.ix_(self.ind_C)] =q_b_C
        
        self.q_b=q_b
        self.q_f=q_f
        
        return  self.q_f, self.q_b 
    

 
    def reac_density(self):
        """
        Given the forward and backward committor and the stationary density, 
        we can compute the normalized density of reactive trajectories, 
        i.e. the probability to be at x in St, given the chain is reactive.
        """
        reac_dens=np.multiply(self.q_b,np.multiply(self.stat_dens,self.q_f))
        self.reac_dens=reac_dens/np.sum(reac_dens) #normalization
        return self.reac_dens
    
    def reac_current(self):
        """
        Computes the reactive current current[i,j] between nodes i and j, as the 
        flow of reactive trajectories from i to j during one time step. 
        """
        current=np.zeros(np.shape(self.P))
        eff_current=np.zeros(np.shape(self.P))
        for i in np.arange(self.S ):
            for j in np.arange(self.S ):
                current[i,j]=self.stat_dens[i]*self.q_b[i]*self.P[i,j]*self.q_f[j]
                if i+1>j:
                    eff_current[i,j]=np.max([0,current[i,j]-current[j,i]])
                    eff_current[j,i]=np.max([0,current[j,i]-current[i,j]])
        self.current=current
        self.eff_current=eff_current
        return self.current, self.eff_current
    
    def transition_rate(self):
        """
        The transition rate is the average flow of reactive trajectories out of A
        """
        self.rate= np.sum(self.current[self.ind_A,:])
        return self.rate
    
    def current_density(self):
        """
        The current density is sum of effective currents over all neighbours of cell.
        """
        current_dens=np.zeros(self.S )
        for i in self.ind_C:
            current_dens[i]=np.sum(self.eff_current[i,:])
        self.current_dens=current_dens
        return self.current_dens
    
    #question what happens if I use a want to compute the rate without having computed the current before? 