import transition_paths as tp
import transition_paths_periodic as tpp
import transition_paths_finite as tpf

from plotting import plot_network_density as plot_density, \
                     plot_network_effective_current as plot_effective_current, \
                     plot_network_effcurrent_and_rate as plot_effcurrent_and_rate, \
                     plot_rate, \
                     plot_reactiveness, \
                     plot_convergence, \
                     plot_colorbar_only

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

# load data about small network
states = np.load(
    os.path.join(data_path, example_name + '_' + 'states.npy'),
    allow_pickle=True, 
)
states = states.item()
labels = np.load(
    os.path.join(data_path, example_name + '_' + 'labels.npy'),
    allow_pickle=True, 
)
labels = labels.item()
pos = np.load(
    os.path.join(data_path, example_name + '_' + 'pos.npy'),
    allow_pickle=True,
)
pos = pos.item()
T = np.load(os.path.join(data_path, example_name + '_' + 'T.npy'))
L = np.load(os.path.join(data_path, example_name + '_' + 'L.npy'))
K = np.load(os.path.join(data_path, example_name + '_' + 'K.npy'))

S = len(states)

ind_A = np.array([0])
ind_C = np.arange(1, np.shape(T)[0] - 1)
ind_B = np.array([4])


# TPT ergodic, infinite-time

# transition matrix
P = T + L

# instantiate
small = tp.transitions_mcs(P, ind_A, ind_B, ind_C)
stat_dens = small.stationary_density()

# compute committor probabilities
[q_f, q_b] = small.committor()

# therof compute the normalized reactive density
norm_reac_dens = small.norm_reac_density()

# and reactive currents
[current, eff_current] = small.reac_current()
rate = small.transition_rate()  # AB discrete transition rate

mean_length = small.mean_transition_length()

#compute share along upper (1) and lower path (via 3)
eff_out = eff_current[0,1]+eff_current[0,3]
share_1 = eff_current[0,1]/eff_out
share_3 = eff_current[0,3]/eff_out
print('In the infinite-time, stationary case, a share of '+str(share_3)+' outflow is via 3, while a share of '+str(share_1)+' outflow is via 1')


# TPT periodisch
# use as transition matrix T + wL, where w varies from 1..0..-1...0
# either faster switching or slower dynamics

M = 6  # 6 size of period

# transition matrix at time k
def P_p(k):
    # varies the transition matrices periodically, by weighting the added
    # matrix L with weights 1..0..-1.. over one period
    return T + np.cos(k*2.*np.pi/M)*L


# instantiate
small_periodic = tpp.transitions_periodic(P_p, M, ind_A, ind_B, ind_C)
stat_dens_p = small_periodic.stationary_density()

[q_f_p, q_b_p] = small_periodic.committor()
P_back_m = small_periodic.backward_transitions()

# normalized reactive density
norm_reac_dens_p = small_periodic.norm_reac_density()

# and reactive currents
[current_p, eff_current_p] = small_periodic.reac_current()

[rate_p, time_av_rate_p] = small_periodic.transition_rate()

mean_length_p = small_periodic.mean_transition_length()


# TPT finite time, time-homogeneous

# transition matrix at time n
def P_hom(n):
    return P

# initial density
init_dens_small = stat_dens
N = 5  # size of time interval

# instantiate
small_finite = tpf.transitions_finite_time(
    P_hom, N, ind_A, ind_B,  ind_C, init_dens_small)
[q_f_f, q_b_f] = small_finite.committor()

stat_dens_f = small_finite.density()

# reactive density (zero at time 0 and N)
reac_norm_factor_f = small_finite.reac_norm_factor()
norm_reac_dens_f = small_finite.norm_reac_density()

# and reactive currents
[current_f, eff_current_f] = small_finite.reac_current()

# first row, out rate of A, second row in rate for B
[rate_f, time_av_rate_f] = small_finite.transition_rate()

mean_length_f = small_finite.mean_transition_length()


# TPT finite time, time-inhomogeneous
# size of time interval
N_inhom = 5 

# transition matrix at time n

def P_inhom(n):
    if np.mod(n,2)==0:
        return P + K
    else: 
        return P - K

def P_inhom_2(n):
    if n in [0, 1, 2, 7, 8, 9]: 
        return P - K/3
    elif n in [3, 6]:
        return P
    else:
        return P + K

def P_inhom_3(n):
    return np.sin(n*2.*np.pi/N_inhom)*K

# initial density
init_dens_small_inhom = stat_dens

# instantiate
small_inhom = tpf.transitions_finite_time(
    P_inhom,
    N_inhom,
    ind_A,
    ind_B,
    ind_C,
    init_dens_small_inhom,
)
[q_f_inhom, q_b_inhom] = small_inhom.committor()

stat_dens_inhom = small_inhom.density()
# reactive density (zero at time 0 and N)
reac_norm_factor_inhom = small_inhom.reac_norm_factor()
norm_reac_dens_inhom = small_inhom.norm_reac_density()

# and reactive currents
[current_inhom, eff_current_inhom] = small_inhom.reac_current()

# first row, out rate of A, second row in rate for B
[rate_inhom, time_av_rate_inhom] = small_inhom.transition_rate()

mean_length_inhom = small_inhom.mean_transition_length()


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

npz_path = os.path.join(data_path, example_name + '_' + 'ergodic.npz')
np.savez(
    npz_path,
    stat_dens=stat_dens,
    q_f=q_f,
    q_b=q_b,
    norm_reac_dens=norm_reac_dens,
    eff_current=eff_current,
    rate=rate,
)
npz_path = os.path.join(data_path, example_name + '_' + 'periodic.npz')
np.savez(
    npz_path,
    stat_dens=stat_dens_p,
    q_f=q_f_p,
    q_b=q_b_p,
    norm_reac_dens=norm_reac_dens_p,
    eff_current=eff_current_p,
    rate=rate_p,
)
npz_path = os.path.join(data_path, example_name + '_' + 'finite.npz')
np.savez(
    npz_path,
    stat_dens=stat_dens_f,
    q_f=q_f_f,
    q_b=q_b_f,
    norm_reac_dens=norm_reac_dens_f,
    reac_norm_factor=reac_norm_factor_f,
    eff_current=eff_current_f,
    rate=rate_f,
    time_av_rate=time_av_rate_f,
)
npz_path = os.path.join(data_path, example_name + '_' + 'inhom.npz')
np.savez(
    npz_path,
    stat_dens=stat_dens_inhom,
    q_f=q_f_inhom,
    q_b=q_b_inhom,
    norm_reac_dens=norm_reac_dens_inhom,
    reac_norm_factor=reac_norm_factor_inhom,
    eff_current=eff_current_inhom,
    rate=rate_inhom,
    time_av_rate=time_av_rate_inhom,
)
npz_path = os.path.join(data_path, example_name + '_' + 'conv.npz')
np.savez(
    npz_path,
    q_f=q_f_conv,
    q_b=q_b_conv,
)
