#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:09:23 2019

@author: bzfhelfm
"""
import numpy as np 

def transitionmatrix_1D(forces,period, sigma, dt, lag, Nstep, interval, dx_power):
    dx=1./(10**dx_power)
    xs = np.round(np.arange(interval[0], interval[1]+dx, dx),dx_power)
    count = np.zeros((len(xs),len(xs),period))
    for j in range(Nstep):
        for p in range(period):
            for seed in range(len(xs)):
                current_X = xs[seed]
                for l in range(lag):
                    new_X = current_X - (forces[p](current_X))*dt + sigma*np.sqrt(dt)*np.random.randn()
                    if new_X <interval[0]: #reflective boundary conditions
                        new_X = interval[0] + (interval[0]-new_X)
                    if new_X >interval[1]:
                        new_X = interval[1] - (new_X - interval[1])
                    current_X=new_X
                from_box = seed  
                to_box  = np.where(xs==round(new_X, dx_power))[0][0]
                count[from_box,to_box,p] += 1.
    return count/Nstep


def transitionmatrix_2D(forces,dim_dV, sigma, dt, lag, Nstep, interval, dx_power,x,y,dim):
    x=np.round(x,dx_power) #dx=1./(10**dx_power)
    y=np.round(y,dx_power)
    xy=[x,y]
    xv, yv = np.meshgrid(x, y)
    
    xdim = np.shape(xv)[1]
    ydim = np.shape(xv)[0]
        
    xn=np.reshape(xv,(xdim*ydim,1))
    yn=np.reshape(yv,(xdim*ydim,1))
    
    grid = np.squeeze(np.array([xn,yn]))
    states_dim = np.shape(grid)[1]
    count = np.zeros((states_dim,states_dim,dim_dV))
    to_box_int=np.zeros(dim)
    
    for j in range(Nstep):
        for p in range(dim_dV):
            for seed in range(states_dim):
                #todo: if cells are large, need to reweigh with stationary density in each cell
                current_X = grid[:,seed]
                for l in range(lag):
                    new_X = current_X - (forces[p](current_X[0],current_X[1]))*dt + sigma*np.sqrt(dt)*np.random.randn(2)
                    for d in range(dim):
                        if new_X[d] <interval[d,0]: #reflective boundary conditions
                            new_X[d] = interval[d,0] + (interval[d,0]-new_X[d])
                        if new_X[d] >interval[d,1]:
                            new_X[d] = interval[d,1] - (new_X[d] - interval[d,1])
                    current_X=new_X
                from_box = seed
                for d in range(dim):
                    to_box_int[d]=np.where(xy[d]==np.round(new_X[d],dx_power))[0][0]
                to_box = np.int(to_box_int[0] + xdim*to_box_int[1])
                count[from_box,to_box,p] += 1.
    return count/Nstep
