#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:28:32 2019

@author: bzfhelfm
"""
import networkx as nx
import numpy as np


def adj_2_transition_matrix(G):
    """
    given a networkx graph, return the transition matrix of the 
    largest connected component (transition probability from i to j if there is an
    edge is 1/deg(i)), also return the subgraph
    """
    largest_cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_cc)
 
    A = nx.adjacency_matrix(H)
    A=A.todense()
    A = np.squeeze(np.asarray(A))
    A=A.astype(float)
    #probability matrix
    T=np.transpose( A/np.sum(A,axis=0))
    
    return H,T

 