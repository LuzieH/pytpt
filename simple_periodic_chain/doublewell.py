#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:02:41 2019

@author: bzfhelfm
"""

import numpy as np
import matplotlib.pyplot as plt
#periodic double well potential in 1D on [-2, 2]

factor=2
V0 = lambda x: factor*(x**2-1)**2
dV0 = lambda x: factor* 4*x*(x**2-1)

V1 = lambda x: factor*(x**2-1)**2 +0.25*x
dV1 =  lambda x: factor*4*x*(x**2-1) + 0.25

V2 = lambda x: factor*(x**2-1)**2 +0.5*x
dV2 = lambda x: factor*4*x*(x**2-1) + 0.5

V3 = lambda x: factor*(x**2-1)**2 + 0.7*x
dV3 = lambda x: factor*4*x*(x**2-1)  + 0.7


#plt.scatter(x,dV(x))

dVs =[dV0, dV1, dV2, dV3, dV2, dV1]
period = 6
interval = [-2,2]
dx_power=2
dx=1./(10**dx_power)
xs = np.round(np.arange(interval[0], interval[1]+dx, dx),dx_power)
dim=np.shape(xs)[0]
Nstep=10000
sigma=0.8 #0.1
dt=0.01
lag=10
#thus the time between steps later will be lag*dt

C = transitionmatrix_1D(dVs,period, sigma, dt, lag,Nstep, interval, dx_power)
P0=np.transpose(C[:,:,0]) #columnsum
P1=np.transpose(C[:,:,1])
P2=np.transpose(C[:,:,2])
P3=np.transpose(C[:,:,3])
P4=np.transpose(C[:,:,4])
P5=np.transpose(C[:,:,5])

#stationary densiites
P_mult = P5.dot(P4.dot(P3.dot(P2.dot(P1.dot(P0)))))
x_0=1./(dim*dx)*np.ones(dim) # uniform starting distribution
Pk=np.linalg.matrix_power(P_mult,10000)
pi0=Pk.dot(x_0) #"periodic" stationary distributions
pi1=P1.dot(pi0)
pi2=P2.dot(pi1)
pi3=P3.dot(pi2)
pi4=P4.dot(pi3)
pi5=P5.dot(pi4)
#committors

qs=periodic_committors(C, A,B,xs)


fig1 = plt.figure()
plt.plot(xs,V0(xs),label='V_0')
plt.plot(xs,V1(xs),label='V_1=V_5')
plt.plot(xs,V2(xs),label='V_2=V_4')
plt.plot(xs,V3(xs),label='V_3')
plt.legend()
plt.title('Potentials')
plt.show()

fig2 = plt.figure()
plt.plot(xs,qs[:,0],label='q_0')
plt.plot(xs,qs[:,1],label='q_1')
plt.plot(xs,qs[:,2],label='q_2')
plt.plot(xs,qs[:,3],label='q_3')
plt.plot(xs,qs[:,4],label='q_4')
plt.plot(xs,qs[:,5],label='q_5')
plt.legend()
plt.title('Committors')
plt.show()



fig3 = plt.figure()
plt.plot(xs,pi0,label='pi_0')
plt.plot(xs,pi1,label='pi_1')
plt.plot(xs,pi2,label='pi_2')
plt.plot(xs,pi3,label='pi_3')
plt.plot(xs,pi4,label='pi_4')
plt.plot(xs,pi5,label='pi_5')
plt.legend()
plt.title('Periodic Stationary Densities')
plt.show()




#fig1 = plt.figure()
#plt.imshow(np.transpose(P0))
#plt.show()
#
#fig2 = plt.figure()
#plt.imshow(np.transpose(P1))
#plt.show()
#
#fig3 = plt.figure()
#plt.imshow(np.transpose(P2))
#plt.show()
#
#fig4 = plt.figure()
#plt.imshow(np.transpose(P3))
#plt.show()

#
#fig5 = plt.figure()
#Pk=np.linalg.matrix_power(np.transpose(P0),50)
#plt.imshow(Pk)
#
#fig6 = plt.figure()
#
#x0=1./(dim*dx)*np.ones(dim)
#xk=Pk.dot(x0)
#plt.scatter(np.arange(dim),xk)
#plt.show()