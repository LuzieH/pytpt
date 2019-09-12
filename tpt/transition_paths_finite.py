#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:48:25 2019

@author: Luzie Helfmann, luziehelfmann@gmail.com
"""

import numpy as np
from inspect import isfunction

class transitions_finite_time:
    """Calculates committor probabilities and transition statistics of 
    Markov chain models over a finite time interval {0,...,N-1} of size N"""

    def __init__(self, P, N, ind_A, ind_B,  ind_C, init_dens):
        """
        Initialize an instance by defining the transition matrix and the sets 
        between which the transition statistics should be computed.
        
        Parameters:
        P: array
            #if the dynamics are time-independent:
            #    irreducible and row-stochastic (rows sum to 1) transition matrix  
            #    of size S x S, S is the size of the state space St={1,2,...,S} 
            if the dynamics are time-dependent: 
                function P(n) is a transition matrix defined for n=0,...,N
        N: int
            size of the time interval {0,1,...,N-1}
        ind_A: array
            set of indices of the state space that belong to the set A
        ind_B: array
            set of indices of the state space that belong to the set B
        ind_C: array
            set of indices of the state space that belong to the transition 
            region C, i.e. the set C = St\(A u B)        
        init_dens: array
            initial density at time 0
        """
 
        assert isfunction(P) == True, "The transition matrices need to inputted \
        as a function mapping time to the corresponding transition matrix."
        
        self._init_dens = init_dens      
        self._ind_A = ind_A
        self._ind_B = ind_B
        self._ind_C = ind_C
        self._N = N
        self._P = P
        self._S = np.shape(P(0))[0] # size of the state space

        self._q_b = None # backward committor
        self._q_f = None # forward committor
        self._reac_dens = None # reactive density
        self._current = None # reactive current
        self._eff_current = None # effective reactive current
        self._rate = None # rate of transitions from A to B
        self._current_dens = None # density of the effective current
        

    def density(self):
        """
        Function that computes and returns an array containing the probability
        to be at time n in node i, the first index of the returned array is time n,
        the second is space/the node i. 
        """
        dens_n = np.zeros((self._N, self._S))
        
        # initial density
        dens_n[0,:] = self._init_dens
        
        for n in np.arange(self._N-1):
            # compute density at next time n+1 by applying the transition matrix
            dens_n[n+1,:] = dens_n[n,:].dot(self._P(n))
            
        return dens_n
    
    def committor(self):
        """
        Function that computes the forward committor q_f (probability that the 
        particle at time n will next go to B rather than A) and backward commitor q_b 
        (probability that the system at time n last came from A rather than B) for all times
        n in {0,...,N-1}
        """
        
        q_f = np.zeros((self._N,self._S))
        q_b = np.zeros((self._N,self._S))
        
        #  forward committor is 1 on B, 0 on A, at time N, q_f is additionally 0 on C
        q_f[self._N-1,self._ind_B] = 1
        #  backward committor is 1 on A, 0 on B, at time 0, q_b is additionally 0 on C
        q_b[0,self._ind_A] = 1
        
        # density at time n-1
        dens_nmin1 = self._init_dens
        
        # iterate through all times n, backward in time for q_f, forward in time for q_b
        for n in range(1,self._N):
            
            # define the restricted transition matrices at time N-n-1
            P_CC = self._P(self._N-n-1)[np.ix_(self._ind_C,self._ind_C)] # entries from C to C
            P_CB = self._P(self._N-n-1)[np.ix_(self._ind_C,self._ind_B)] # entries from C to B
            
            # compute forward committor backwards in time
            q_f[self._N-n-1,self._ind_C]= P_CC.dot(q_f[self._N-n,self._ind_C]) \
                + P_CB.dot(np.ones(np.size(self._ind_B)))
                
            # forward committor is 1 on B, 0 on A
            q_f[self._N-n-1,self._ind_B] = 1
            
            # density at time n
            dens_n = dens_nmin1.dot(self._P(n-1))
            
            # ensure that when dividing by the distribution and it's negative, 
            # we don't divide by zero, there is no contribution, thus we can replace
            # the inverse by zero
            d_n_inv = dens_n[self._ind_C]
            for i in range(np.size(self._ind_C)):
                if d_n_inv[i]>0:
                    d_n_inv[i] =1/d_n_inv[i] 
                # else: its just zero
            
            # define restricted transition matrices
            P_CC = self._P(n)[np.ix_(self._ind_C,self._ind_C)]
            P_CA = self._P(n)[np.ix_(self._ind_A,self._ind_C)]
            
            # compute backward committor forward in time 
            q_b[n,self._ind_C] =  d_n_inv*(dens_nmin1[self._ind_C]*q_b[n-1,self._ind_C]).dot(P_CC) \
                + d_n_inv*dens_nmin1[self._ind_A].dot(P_CA)
                
            # backward committor is 1 on A, 0 on B
            q_b[n,self._ind_A] = 1
            
            dens_nmin1  = dens_n
     
        self._q_b = q_b
        self._q_f = q_f
        
        return  self._q_f, self._q_b  

    def reac_density(self):
        """
        Given the forward and backward committor and the density, 
        we can compute the normalized density of reactive trajectories, 
        i.e. the probability to be at x in S at time n, given the chain is reactive.
        The function returns an array of the reactive density for each time 
        (with time as the first index of the array).
        """
        assert self._q_f.all() != None, "The committor functions need \
        first to be computed by using the method committor"
        
        reac_dens = np.zeros((self._N,self._S))   

        # density at time n, start with n=1        
        dens_n = self._init_dens.dot(self._P(0))
        
        # at time 0 and N+1, the reactive density is zero, the event "to be reactive" is not possiböe
        for n in range(1,self._N-1):
            
            reac_dens[n,:] = np.multiply(self._q_b[n,:],np.multiply(dens_n,self._q_f[n,:]))
            reac_dens[n,:] = reac_dens[n,:]/np.sum(reac_dens[n,:]) # normalization
            
            # update density for next time point
            dens_n = dens_n.dot(self._P(n-1))
        
        self._reac_dens = reac_dens
        
        return self._reac_dens
    
    def reac_current(self):
        """
        Computes the reactive current current[i,j] between nodes i and j, as the 
        flow of reactive trajectories from i to j during one time step. 
        """
        assert self._q_f.all() != None, "The committor functions  need \
        first to be computed by using the method committor"
        
        current = np.zeros((self._N,np.shape(self._P(0))[0],np.shape(self._P(0))[0]))
        eff_current = np.zeros((self._N,np.shape(self._P(0))[0],np.shape(self._P(0))[0]))
        
        dens_n = self._init_dens
        
        for n in range(self._N):
        
            for i in np.arange(self._S ):
                for j in np.arange(self._S ):
                    current[n,i,j] = dens_n[i]*self._q_b[n,i]*self._P(n)[i,j]*self._q_f[n,j]
                    
                    if i+1>j:
                        eff_current[n,i,j] = np.max([0,current[n,i,j]-current[n,j,i]])
                        eff_current[n,j,i] = np.max([0,current[n,j,i]-current[n,i,j]])
                        
            dens_n = dens_n.dot(self._P(n))
                        
        self._current = current
        self._eff_current = eff_current
        
        return self._current, self._eff_current
    
    def transition_rate(self):
        
        """
        The transition rate is the average flow of reactive trajectories out of A
        """
       
        assert self._current.all() != None, "The reactive current first needs \
        to be computed by using the method reac_current"
        
        self._rate =  np.sum(self._current[:,self._ind_A,:],axis = (1,2))

        return self._rate
    
    def current_density(self):
        """
        The current density in a node is the sum of effective currents 
        over all neighbours of the node.
        """
        
        assert self._current.all() != None, "The reactive current first needs \
        to be computed by using the method reac_current"
        
        current_dens = np.zeros((self._N,self._S))
        
        for n in range(self._N):
            for i in self._ind_C:
                current_dens[n,i] = np.sum(self._eff_current[n,i,:])
            self._current_dens = current_dens
            
        return self._current_dens