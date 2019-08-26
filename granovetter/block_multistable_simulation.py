#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:18:52 2019

@author: bzfhelfm
"""
from networkx import from_numpy_matrix, draw
import numpy as np
import Granovetter 
import stochastic_block_model
import matplotlib.pyplot as plt
import os.path


load_data=True

if load_data:
    #########################################
    #load
     
    my_path = os.path.abspath(os.path.dirname(__file__))
    A = np.load(os.path.join(my_path, 'networks/A2.npy'))
    G = from_numpy_matrix(A)

else: 
   #########################################
    #Create block model graph
    
    #cluster sizes
    sizes = np.array([50,50,50,50,50])
    #symmetric wiring probabiliies between clusters
    probs = np.array([[0.1,0.02,0,0.01,0],
             [0.02,0.1,0.02,0,0],
             [0,0.02,0.1,0,0.02],
             [0.01,0,0,0.1,0.01],
             [0,0,0.02,0.01,0.1]])
    
    A = stochastic_block_model.adjacency(sizes, probs)
    #create networkx graph
    
    G = from_numpy_matrix(A)
    #save graph
    np.save('/nfs/numerik/bzfhelfm/Dokumente/PhD/code/transitions/granovetter/networks/A3',A )
    
plt.figure()
draw(G)


########################################
#Granovetter stochastic micro model

#initial conditions
active_nodes=np.arange(40)
inactive_nodes=np.arange(210,250)
#initially active nodes drawn randomly from the remaining nodes
jrandom = np.random.choice(np.arange(0,170),1)[0] #random number of initially active nodes
initially_active_nodes= np.random.choice(np.arange(40,210),jrandom,replace=False)
initially_inactive_nodes=np.array([i for i in np.arange(40,210) if i not in initially_active_nodes])

#micromodel instanciation
block=Granovetter.micromodel_stochastic(A,  active_nodes, inactive_nodes, initially_active_nodes, initially_inactive_nodes,e_active=0.1,e_inactive=0.1,p_active=0.9,p_inactive=0.9)#e_active=0.05,e_inactive=0.05,p_active=0.8,p_inactive=0.8)



#number of time steps
maxtime=100000
#run simulation
R=block.run(maxtime)

#plotting
plt.figure()
plt.plot(R)

plt.figure()
plt.hist(R,250, range=(0,250))
#plt.scatter(R[:maxtime+1],R[1:])
#plt.plot(np.array([40,210]),np.array([40,210]))
#plt.xlim(0,250)
#plt.ylim(0,250)

