#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:59:08 2019

@author: bzfhelfm
"""

import networkx as nx
import numpy as np
import os.path
import stoch_networks as sn
import transition_paths as tp


my_path = os.path.abspath(os.path.dirname(__file__))

##################################################
##adjacency matrix 
#A=np.zeros((10,10))
#A[0,3]=1
#A[3,4]=1
#A[4,5]=1
#A[5,2]=1
#A[2,1]=1
#A[1,8]=1
#A[8,7]=1
#A[7,6]=1
#A[6,3]=1
#A[7,9]=1
#A[8,9]=1
#A=A+A.transpose()
#
##node positions
#pos = {0: (0,0), 1: (4,-0.5), 2: (4,0), 3: (1,0), 4: (2,0), 5: (3, 0), 6: (1, -0.5), 7: (2, -1), 8: (3,-1) ,9: (2.5, -0.5) }
#G = nx.from_numpy_matrix(A)
#
###########################################
## transition matrix
#T = np.zeros((10,10))
#T[0,3]=0.3
#T[0,0]=0.7
#T[3,4]=0.2
#T[4,5]=0.4
#T[5,2]=0.5
#T[2,2]=0.8
#T[2,1]=0.1
#T[1,8]=0.1
#T[1,1]=0.8
#T[8,7]=0.5
#T[7,6]=0.1
#T[7,7]=0.6
#T[6,3]=0.4
#T[3,0]=0.6
#T[4,3]=0.6
#T[5,4]=0.5
#T[2,5]=0.1
#T[1,2]=0.1
#T[8,1]=0.1
#T[8,8] = 0.1
#T[7,8]=0.1
#T[6,7]=0.2
#T[6,6]=0.4
#T[3,6]=0.2
#T[9,9]=0.8
#T[7,9] = 0.2
#T[9,7] = 0.1
#T[8,9] = 0.3
#T[9,8] = 0.1

#################################################
#adjacency matrix 
A=np.zeros((9,9))
A[0,3]=1
A[3,4]=1
A[4,5]=1
A[5,1]=1
A[1,8]=1
A[8,7]=1
A[7,6]=1
A[6,3]=1
A[7,2]=1
A[2,4]=1
A[2,8]=1
A[2,5]=1
A=A+A.transpose()

#node positions
pos = {6: (1,-1), 1: (3,-0.5), 2: (2,-0.5), 0: (1,0), 4: (2,0), 5: (3, 0), 3: (1, -0.5), 7: (2, -1), 8: (3,-1) }

# some math labels
labels={0: r'$A$', 1: r'$B$', 2:'2', 3:'3', 4:'4', 5:'5',6:'6',7:'7',8:'8'}

G = nx.from_numpy_matrix(A)

##########################################
# transition matrix
T = np.zeros((9,9))
T[0,3]=0.3
T[0,0]=0.7
T[3,4]=0.1
T[4,5]=0.3
T[5,1]=0.6
T[1,8]=0.1
T[1,1]=0.8
T[8,7]=0.3
T[8,8]=0.4
T[7,6]=0.1
T[7,7]=0.3
T[6,3]=0.3
T[3,0]=0.6
T[4,3]=0.7
T[5,4]=0.4
T[1,5]=0.1
T[8,1]=0.3
T[7,8]=0.1
T[6,7]=0.3
T[6,6]=0.4
T[3,6]=0.3
T[2,2]=0.8
T[2,7]=0.1
T[7,2] = 0.5
T[2,8] = 0.1


##############for the periodic setting, add the following matrix with weights varying between 0 and 1
T_periodisch= np.zeros(np.shape(T))
T_periodisch[7,2]=-0.5
T_periodisch[7,8]=0.5
T_periodisch[2,8]=-0.1
T_periodisch[2,5]=0.1
T_periodisch[4,2]=0.5
T_periodisch[4,5]=-0.2
T_periodisch[4,3]=-0.3
T_periodisch[3,4]=0.2
T_periodisch[3,6]=-0.2
############################################

np.save(os.path.join(my_path,'networks_data/small_network_A.npy'), A)
np.save(os.path.join(my_path,'networks_data/small_network_T.npy'), T)
np.save(os.path.join(my_path,'networks_data/small_network_pos.npy'), pos)
np.save(os.path.join(my_path,'networks_data/small_network_labels.npy'), labels)
np.save(os.path.join(my_path,'networks_data/small_network_T_periodisch.npy'), T_periodisch)