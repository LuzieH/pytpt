#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:08:38 2019

@author: bzfhelfm
"""
#import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



##########################################committor animation

savename = '2D_committors'
number_frames = 10*period
min_value=0
max_value=1


def anim_fun_comm(i):
    return np.reshape(qs[:,i],(xdim,ydim))


# density animation
    
savename2 = '2D_stat_dens'
number_frames2 = 10*period
min_value2=0
max_value2=np.max(pis)


def anim_fun_stat_dens(i):
    return np.reshape(pis[:,i],(xdim,ydim))

#anim_2D(anim_fun_comm,savename,min_value, max_value, number_frames,period)    
#anim_2D(anim_fun_stat_dens,savename2,min_value2, max_value2, number_frames2,period)  

## density animation
#    
#savename3 = '2D_reac_dens'
#number_frames3 = 10*period
#min_value3=0
#max_value3=np.max(reac_pis)


def anim_fun_reac_dens(i):
    return np.reshape(reac_pis[:,i],(xdim,ydim))
################################################
    

def anim_2D(anim_fun2,savename,min_value, max_value, number_frames,period):
    
    fig = plt.figure()
    
    i=0
    im = plt.imshow(anim_fun2(i), animated=True, vmin=min_value, vmax=max_value)
    
    def updatefig(i):
       # global i
        i+=1
        im.set_array(anim_fun2(np.mod(i,period)))
        return im,
    
    ani = animation.FuncAnimation(fig, updatefig, frames=number_frames-1, interval=200, repeat=True, blit=True)
    
    ani.save(savename+'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    #plt.show()
    
