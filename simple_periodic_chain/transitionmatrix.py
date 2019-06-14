#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:09:23 2019

@author: bzfhelfm
"""
import numpy as np

#def transitionmatrix_1D(forces,period, sigma, dt, Nstep, interval, dx_power,start_X):
#    dx=1./(10**dx_power)
#    xs = np.round(np.arange(interval[0], interval[1]+dx, dx),dx_power)
#    count = np.zeros((len(xs),len(xs),period))
#    current_X = start_X
#    traj=np.zeros((Nstep,period))
#    for j in range(Nstep):
#        for p in range(period):
#            new_X = current_X - forces[p](current_X)*dt + sigma* np.sqrt(dt)*np.random.randn()
#            if new_X <interval[0]:
#                new_X = interval[0]
#            if new_X >interval[1]:
#                new_X = interval[1]
#            from_box = np.where(xs==round(current_X, dx_power))[0][0]
#            to_box  = np.where(xs==round(new_X, dx_power))[0][0]
#            count[from_box,to_box,p] += 1
#            current_X = new_X
#            traj[j,p]=current_X
#    
#    return count,traj

def transitionmatrix_1D(forces,period, sigma, dt, Nstep, interval, dx_power):
    dx=1./(10**dx_power)
    xs = np.round(np.arange(interval[0], interval[1]+dx, dx),dx_power)
    count = np.zeros((len(xs),len(xs),period))
    
    for j in range(Nstep):
        for p in range(period):
            for seed in range(len(xs)):
                current_X = xs[seed]
                new_X = current_X - forces[p](current_X)*dt + sigma* np.sqrt(dt)*np.random.randn()
                if new_X <interval[0]:
                    new_X = interval[0]
                if new_X >interval[1]:
                    new_X = interval[1]
                from_box = seed #np.where(xs==round(current_X, dx_power))[0][0]
                to_box  = np.where(xs==round(new_X, dx_power))[0][0]
                count[from_box,to_box,p] += 1
    
    return count