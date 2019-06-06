#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:02:41 2019

@author: bzfhelfm
"""

import numpy as np
import matplotlib.pyplot as plt
#periodic double well potential in 1D on [-2, 2]
V1 = lambda x: (x**2-1)**2
dV1 = lambda x: 4*x*(x*+2-1)

V2 = lambda x: (x**2-1)**2 +0.25*x
dV2 = lambda x: 4*x*(x*+2-1) + 0.25

V3 = lambda x: (x**2-1)**2 + 0.5*x
dV3 = lambda x: 4*x*(x*+2-1) + 0.5


x = np.arange(-2,2,0.1)
plt.scatter(x,V1(x))
plt.scatter(x,V2(x))
plt.scatter(x,V3(x))
#plt.scatter(x,dV(x))

dVs = [V1, V2, V3, V2]
period = 4
interval = [-2, 2]
dx_power=1
dx=1./(10**dx_power)


C = transitionmatrix_1D(dVs,period, 0.1, 0.1, 1000, interval, dx_power,0)