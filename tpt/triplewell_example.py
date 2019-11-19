import transition_paths as tp
import transition_paths_periodic as tpp
import transition_paths_finite as tpf
 
import numpy as np
import matplotlib.pyplot as plt

import os.path
 
##############################################################################
# load data
 
my_path = os.path.abspath(os.path.dirname(__file__))
T = np.load(os.path.join(my_path, 'data/triplewell_T.npy'))
T_m = np.load(os.path.join(my_path, 'data/triplewell_T_m.npy'))
interval = np.load(os.path.join(my_path, 'data/triplewell_interval.npy'))
dx = np.load(os.path.join(my_path, 'data/triplewell_dx.npy'))
ind_A = np.load(os.path.join(my_path, 'data/triplewell_ind_A.npy'))
ind_B = np.load(os.path.join(my_path, 'data/triplewell_ind_B.npy'))
ind_C = np.load(os.path.join(my_path, 'data/triplewell_ind_C.npy'))

############################################################################
#state space
x = np.arange(interval[0,0],interval[0,1]+dx, dx) #box centers in x and y direction
y = np.arange(interval[1,0],interval[1,1]+dx, dx)
xv, yv = np.meshgrid(x, y)

xdim = np.shape(xv)[0] #discrete dimension in x and y direction
ydim = np.shape(xv)[1]
dim_st = xdim*ydim # dimension of the statespace
xn=np.reshape(xv,(xdim*ydim,1))
yn=np.reshape(yv,(xdim*ydim,1))
grid = np.squeeze(np.array([xn,yn]))


#############################################################################
# infinite-time ergodic

# instantiate
well3 = tp.transitions_mcs(T, ind_A, ind_B, ind_C)
stat_dens = well3.stationary_density()

# compute committor probabilities
[q_f, q_b] = well3.committor()

# therof compute the reactive density
reac_dens = well3.reac_density()

# and reactive currents
[current, eff_current] = well3.reac_current()
rate = well3.transition_rate()  # AB discrete transition rate



#############################################################################
# plots 
plt.figure(1)
plt.imshow(stat_dens.reshape((51,41)))
plt.figure(2)
plt.imshow(reac_dens.reshape((51,41)))
plt.figure(3)
plt.imshow(q_f.reshape((51,41)))
plt.figure(4)
plt.imshow(q_b.reshape((51,41)))


#calculation the effective vector for each state
eff_vectors = np.zeros((dim_st, 2))
for i in np.arange(dim_st):
    for j in np.arange(dim_st):
        eff_vectors[i,0] += eff_current[i,j] *  (xn[j] - xn[i])  
        eff_vectors[i,1] += eff_current[i,j] *  (yn[j] - yn[i])  

plt.quiver(xn,yn,list(eff_vectors[:,0]),list(eff_vectors[:,1]))

#############################################################################
# periodic
M=np.shape(T_m)[0]

def Tm(m): 
    return T_m[np.mod(m,M),:,:].squeeze()

# instantiate
well3_periodic = tpp.transitions_periodic(Tm, M, ind_A, ind_B, ind_C)
stat_dens_p = well3_periodic.stationary_density()

[q_f_p, q_b_p] = well3_periodic.committor()
P_back_m = well3_periodic.backward_transitions()
# reactive density
reac_dens_p = well3_periodic.reac_density()

# and reactive currents
[current_p, eff_current_p] = well3_periodic.reac_current()

rate_p = well3_periodic.transition_rate()