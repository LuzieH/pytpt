import transition_paths as tp
import transition_paths_periodic as tpp
import transition_paths_finite as tpf

import pickle
import numpy as np
import networkx as nx

import os.path

# TODO add colorbar to plots

# general

# define directories path to save the data and figures 
my_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(my_path, 'data')
charts_path = os.path.join(my_path, 'charts')
example_name = 'small_network'

# load small network construction data
network_construction = np.load(
    os.path.join(data_path, example_name + '_' + 'construction.npz'),
    allow_pickle=True,
)
states = network_construction['states'].item()
labels = network_construction['labels'].item()
pos = network_construction['pos'].item()
T = network_construction['T']
L = network_construction['L']
K = network_construction['K']

S = len(states)

ind_A = np.array([0])
ind_C = np.arange(1, np.shape(T)[0] - 1)
ind_B = np.array([4])


# TPT ergodic, infinite-time
dynamics = 'ergodic'
# transition matrix
P = T + L
# instantiate
small = tp.transitions_mcs(P, ind_A, ind_B, ind_C)
# compute statistics
small.compute_statistics()
# save statistics
small.save_statistics(example_name, dynamics)

#compute share along upper (1) and lower path (via 3)
eff_current = small._eff_current
eff_out = eff_current[0, 1] + eff_current[0, 3]
share_1 = eff_current[0, 1] / eff_out
share_3 = eff_current[0, 3] / eff_out
print('In the infinite-time, stationary case, a share of ' + str(share_3) + ' outflow is via 3, while a share of '+str(share_1)+' outflow is via 1')


# TPT periodisch
# use as transition matrix T + wL, where w varies from 1..0..-1...0
# either faster switching or slower dynamics

dynamics = 'periodic'
M = 6  # 6 size of period

# transition matrix at time k
def P_p(k):
    # varies the transition matrices periodically, by weighting the added
    # matrix L with weights 1..0..-1.. over one period
    return T + np.cos(k*2.*np.pi/M)*L

# instantiate
small_periodic = tpp.transitions_periodic(P_p, M, ind_A, ind_B, ind_C)
# compute statistics
small_periodic.compute_statistics()
# save statistics
small_periodic.save_statistics(example_name, dynamics)


# TPT finite time, time-homogeneous
dynamics = 'finite'

# transition matrix at time n
def P_hom(n):
    return P

# initial density
init_dens_small = small.stationary_density()

N = 5  # size of time interval

# instantiate
small_finite = tpf.transitions_finite_time(
    P_hom, N, ind_A, ind_B,  ind_C, init_dens_small)
# compute statistics
small_finite.compute_statistics()
# save statistics
small_finite.save_statistics(example_name, dynamics)


# TPT finite time, time-inhomogeneous
dynamics = 'inhom'
# size of time interval
N_inhom = 5 

# transition matrix at time n

def P_inhom(n):
    if np.mod(n, 2) == 0:
        return P + K
    else: 
        return P - K

def P_inhom_2(n):
    if n in [0, 1, 2, 7, 8, 9]: 
        return P - K / 3
    elif n in [3, 6]:
        return P
    else:
        return P + K

def P_inhom_3(n):
    return np.sin(n*2.*np.pi/N_inhom)*K

# initial density
init_dens_small_inhom = small.stationary_density()

# instantiate
small_inhom = tpf.transitions_finite_time(
    P_inhom,
    N_inhom,
    ind_A,
    ind_B,
    ind_C,
    init_dens_small_inhom,
)
# compute statistics
small_inhom.compute_statistics()
# save statistics
small_inhom.save_statistics(example_name, dynamics)


# TPT finite time extension to infinite time, convergence analysis
N_max = 150  # max value of N
q_f_conv = np.zeros((N_max, S))
q_b_conv = np.zeros((N_max, S))

for n in np.arange(1, N_max + 1):
    # extended time interval
    N_ex = n*2 + 1

    # instantiate
    small_finite_ex = tpf.transitions_finite_time(
        P_hom,
        N_ex,
        ind_A,
        ind_B,
        ind_C,
        init_dens_small,
    )
    
    # compute statistics
    [q_f_ex, q_b_ex] = small_finite_ex.committor()
    q_f_conv[n-1, :] = q_f_ex[n, :]
    q_b_conv[n-1, :] = q_b_ex[n, :]

# save the transition statistics in npz files
npz_path = os.path.join(data_path, example_name + '_' + 'conv.npz')
np.savez(
    npz_path,
    q_f=q_f_conv,
    q_b=q_b_conv,
)
