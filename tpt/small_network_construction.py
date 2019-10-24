import networkx as nx
import numpy as np
import os.path


my_path = os.path.abspath(os.path.dirname(__file__))
 

#################################################
#adjacency matrix 
A=np.zeros((5,5))
A[0,2] = 1
A[0,4] = 1
A[2,3] = 1 
A[3,4] = 1
A[2,3] = 1
A[1,4] = 1
A[1,2] = 1
A=A+A.transpose()

#node positions
pos = {}
pos[0] = (0,-1)
pos[1] = (1,0)
pos[2] = (0,0)
pos[3] = (0.5,-0.5)
pos[4] = (1,-1)

# some math labels
labels={0: r'$A$', 1: r'$B$', 2:'2', 3:'3', 4:'4'}

G = nx.from_numpy_matrix(A)

##########################################
# transition matrix
T = np.zeros((5,5))
#symmetric 
T[0,0] = 0.7
T[0,2] = 0.15
T[2,0] = 0.3
T[0,4] = 0.15
T[4,0] = 0.3
T[2,1] = 0.3
T[1,2] = 0.15
T[1,1] = 0.7
T[1,4] = 0.15
T[4,1] = 0.3
T[3,4] = 0.2
T[3,3] = 0.6
T[4,3] = 0.4
T[2,3] = 0.4
T[3,2] = 0.2


##############for the periodic setting, add the following matrix with weights varying between 0 and 1
T_p = np.zeros(np.shape(T))

#not symmetric, dynamic varies between T+T_p ....T-T_p 
 
 
T_p[0,2] = -0.05
T_p[2,0] = 0.2
T_p[0,4] = 0.05
T_p[4,0] = - 0.2
T_p[2,1] = 0.2
T_p[1,2] = - 0.05
T_p[1,4] = 0.05
T_p[4,1] = - 0.2
T_p[3,4] = 0.2
T_p[4,3] = 0.4
T_p[2,3] = - 0.4
T_p[3,2] = - 0.2
 
############################################

np.save(os.path.join(my_path,'data/small_network_A.npy'), A)
np.save(os.path.join(my_path,'data/small_network_T.npy'), T)
np.save(os.path.join(my_path,'data/small_network_pos.npy'), pos)
np.save(os.path.join(my_path,'data/small_network_labels.npy'), labels)
np.save(os.path.join(my_path,'data/small_network_T_periodisch.npy'), T_p )

################slower transition matrix

factor = 0.5
T_new = factor*T + (1-factor)*np.diag(np.ones(5))

T_p_new = factor*T_p