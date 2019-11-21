import networkx as nx
import numpy as np

import os.path

my_path = os.path.abspath(os.path.dirname(__file__))

# node positions
pos = {}
pos[0] = (0, -1)
pos[1] = (0, 0)
pos[2] = (0.5, -0.5)
pos[3] = (1, -1)
pos[4] = (1, 0)

# some math labels
labels = {
    0: r'$A$',
    1: '1',
    2: '2',
    3: '3',
    4: r'$B$',
}

# T: symmetric stochastic matrix
T = np.zeros((5, 5))
T[0, 0] = 0.7
T[0, 1] = 0.15
T[0, 3] = 0.15
T[1, 0] = 0.3
T[1, 2] = 0.4
T[1, 4] = 0.3
T[2, 1] = 0.2
T[2, 2] = 0.6
T[2, 3] = 0.2
T[3, 0] = 0.3
T[3, 2] = 0.4
T[3, 4] = 0.3
T[4, 1] = 0.15
T[4, 3] = 0.15
T[4, 4] = 0.7

# L: 0-rowsum matrix
# L+T does not have the 1-2 connection and L-T does not have the 2-3 connection 
L = np.zeros(np.shape(T))
L[0, 1] = -0.05
L[0, 3] =  0.05
L[1, 0] =  0.2
L[1, 2] = -0.4
L[1, 4] =  0.2
L[2, 1] = -0.2
L[2, 3] =  0.2
L[3, 0] = -0.2
L[3, 2] =  0.4
L[3, 4] = -0.2
L[4, 1] = -0.05
L[4, 3] =  0.05

# K: 0-rowsum matrix
# L+T+K transition matrix where A and B are less metastable
K = np.zeros((5, 5))
K[0, 0] = -0.2
K[0, 1] =  0.1
K[0, 3] =  0.1
K[4, 1] =  0.1
K[4, 3] =  0.1
K[4, 4] = -0.2

# T+L : transition matrix for the ergodic, infinite-time case
# T+L .... T-L : transition matricec within a period for the periodic case
# L+T : transition matrix for the finite-time, time-homogeneous case 
# L+T+3K, L+T+2K, L+T+K, L+T, ..., L+T : transition matrices for the finite-time,
# time-inhomogeneous case

np.save(os.path.join(my_path, 'data/small_network_pos.npy'), pos)
np.save(os.path.join(my_path, 'data/small_network_labels.npy'), labels)
np.save(os.path.join(my_path, 'data/small_network_T.npy'), T)
np.save(os.path.join(my_path, 'data/small_network_L.npy'), L)
np.save(os.path.join(my_path, 'data/small_network_K.npy'), K)

# slower transition matrix
#
#factor = 0.5
#T_new = factor*T + (1-factor)*np.diag(np.ones(5))
#
#T_p_new = factor*T_p
