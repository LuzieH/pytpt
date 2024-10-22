import numpy as np

import os.path

my_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(my_path, 'data')
example_name = 'small_network'

# classification of states
states = {
    0: 'A',
    1: 'C',
    2: 'C',
    3: 'C',
    4: 'B',
}

# labelling of states
labels = {
    0: r'$A$',
    1: '1',
    2: '2',
    3: '3',
    4: r'$B$',
 }

# position of states
pos = {
    0: (0, 0),
    1: (0, 0.5),
    2: (0.25, 0.25),
    3: (0.5, 0),
    4: (0.5, 0.5),
}

# number of states
S = len(states)

# T: symmetric stochastic matrix
T = np.zeros((S, S))
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
L = np.zeros((S, S))
L[0, 1] =  0.05
L[0, 3] = -0.05
L[1, 0] = -0.2
L[1, 2] =  0.4
L[1, 4] = -0.2
L[2, 1] =  0.2
L[2, 3] = -0.2
L[3, 0] =  0.2
L[3, 2] = -0.4
L[3, 4] =  0.2
L[4, 1] =  0.05
L[4, 3] = -0.05

# K: 0-rowsum matrix
# L+T+K transition matrix where A and B are less metastable
K = np.zeros((S, S))
K[0, 0] = -0.3
K[0, 1] =  0.2
K[0, 3] =  0.1
K[3, 0] =  0.45
K[3, 4] = -0.45
K[4, 1] =  0.2
K[4, 3] =  0.1
K[4, 4] = -0.3

# save construction
npz_path = os.path.join(data_path, example_name + '_' + 'construction.npz')
np.savez(
    npz_path,
    states=states,
    labels=labels,
    pos=pos,
    T=T,
    L=L,
    K=K,
)
