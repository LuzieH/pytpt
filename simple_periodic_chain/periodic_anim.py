#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:31:29 2019

@author: bzfhelfm
"""




"""
=========================
Simple animation examples
=========================

"""

#%matplotlib qt


#def periodic_anim(C, period, maxK, x0, interval,dx,dx_power):
#    fig1 = plt.figure()
#    
#    #x = np.arange(-9, 10)
#    #y = np.arange(-9, 10).reshape(-1, 1)
#    #base = np.hypot(x, y)
#    xk=x0
#    xs = np.round(np.arange(interval[0], interval[1]+dx, dx),dx_power)
#    
#    ims = []
#    for add in np.arange(maxK*period):
#        for t in np.arange(maxK):
#            for p in np.arange(period):
#                Pk=np.transpose(C[:,:,p])
#                xk=Pk.dot(xk)
#                ims.append((plt.scatter(xs, xk)))
#    
#    im_ani = animation.ArtistAnimation(fig1, ims, interval=50, repeat_delay=3000,
#                                       blit=True)
#    
#    # To save this second animation with some metadata, use the following command:
#    im_ani.save('im_periodic_well.mp4')
#    
#    #plt.show()
    
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


maxK=10
xlimit=(interval[0],interval[1])
ylimit=(-0.2,5)
frameslen=maxK*period
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=xlimit, ylim=ylimit)
line, = ax.plot([], [], lw=2)


xs = np.round(np.arange(interval[0], interval[1]+dx, dx),dx_power)
dim=np.shape(xs)[0]
x0=1./(dx*dim)*np.ones(dim)
xks=np.zeros((np.shape(x0)[0],maxK*period))
xks[:,0]=x0
for i in np.arange(maxK*period-1):
    Pk=np.transpose(C[:,:,np.mod(i,period)])
    xks[:,i+1]=Pk.dot(xks[:,i])
        
        
def periodic_anim(xlimit, ylimit,frameslen,xks):


        
    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,
    
    # animation function.  This is called sequentially
    def animate(i):
        line.set_data(xs, xks[:,i])
        return line,
    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frameslen, interval=100, blit=True)
    
    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save('periodic_anim.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    
    plt.show()



####################################################
#
#maxK=10
#
## First set up the figure, the axis, and the plot element we want to animate
#fig = plt.figure()
#ax = plt.axes(xlim=(interval[0],interval[1]), ylim=(-0.2,5))
#line, = ax.plot([], [], lw=2)
#
#
#xs = np.round(np.arange(interval[0], interval[1]+dx, dx),dx_power)
#dim=np.shape(xs)[0]
#x0=1./(dx*dim)*np.ones(dim)
#xks=np.zeros((np.shape(x0)[0],maxK*period))
#xks[:,0]=x0
#for i in np.arange(maxK*period-1):
#    Pk=np.transpose(C[:,:,np.mod(i,period)])
#    xks[:,i+1]=Pk.dot(xks[:,i])
#    
## initialization function: plot the background of each frame
#def init():
#    line.set_data([], [])
#    return line,
#
## animation function.  This is called sequentially
#def animate(i):
#    line.set_data(xs, xks[:,i])
#    return line,
#
## call the animator.  blit=True means only re-draw the parts that have changed.
#anim = animation.FuncAnimation(fig, animate, init_func=init,
#                               frames=maxK*period, interval=100, blit=True)
#
## save the animation as an mp4.  This requires ffmpeg or mencoder to be
## installed.  The extra_args ensure that the x264 codec is used, so that
## the video can be embedded in html5.  You may need to adjust this for
## your system: for more information, see
## http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('periodic_anim.mp4', fps=30, extra_args=['-vcodec', 'libx264'])