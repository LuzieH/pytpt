#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:03:18 2019

@author: bzfhelfm
"""
import numpy as np
import transition_paths_finite as tpf
import matplotlib.pyplot as plt

#P = lambda n: P
def Pn(n):
    return P

N=10
ind_A=np.arange(50)
ind_B=np.arange(200,250)
ind_C=np.arange(50,200)
init_dens=1./250*np.ones(250)
model=tpf.transitions_finite_time(Pn, N, ind_A, ind_B,  ind_C, init_dens)
[q_f,q_b]=model.committor()


