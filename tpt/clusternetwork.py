#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:44:20 2019

@author: bzfhelfm
"""

import networkx as nx
import numpy as np
import os.path

#meshgrid of statdens
x = np.arange(-2, 2+0.1, 0.1)
y = np.arange(-2.5, 2.5+0.1, 0.1)
xx, yy = np.meshgrid(x, y)
xx_fl=xx.flatten()
yy_fl=yy.flatten()

#construct a spatial graph from given positions
n =300
#distribution from which to draw positions
my_path = os.path.abspath(os.path.dirname(__file__))
dist = np.load(os.path.join(my_path,'networks_data/dist.npy'))
samples= np.random.choice(2091,n,p=dist)
pos = {i: (xx_fl[samples[i]], yy_fl[samples[i]]) for i in range(n)}
G = nx.random_geometric_graph(n, 0.25, pos=pos)


#nx.draw(G,pos=pos)
#
#largest_cc = max(nx.connected_components(G), key=len)
#H = G.subgraph(largest_cc)
#nx.draw(H,pos=pos)
#
#
#A = nx.adjacency_matrix(H)
#A=A.todense()
#A = np.squeeze(np.asarray(A))
#A=A.astype(float)
##probability matrix
#T=np.transpose( A/np.sum(A,axis=0))

#connected subgraph induces a transition matrix
[H,T]=adj_2_transition_matrix(G)

#stationary density of the transition matrix
stat_dens_H=stat_dens(T)

nx.draw(H, pos=pos, node_color=stat_dens_H)