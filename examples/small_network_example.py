from pytpt import stationary  
from pytpt import periodic  
from pytpt import finite  
  
import numpy as np
 
import os.path
 
# general

# define directories path to save the data and figures 
my_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(my_path, 'data')
charts_path = os.path.join(my_path, 'charts')

# load small network construction data
network_construction = np.load(
    os.path.join(data_path, 'small_network_construction.npz'),
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


# TPT for the ergodic, infinite-time case
example_name = 'small_network_stationary'
# transition matrix
P = T + L
# instantiate
small = stationary.tpt(P, ind_A, ind_B, ind_C)
# compute statistics
small.compute_statistics()
# save statistics
npz_path = os.path.join(data_path, example_name + '.npz')
small.save_statistics(npz_path)

#compute share along upper (1) and lower path (via 3)
eff_current = small._eff_current
eff_out = eff_current[0, 1] + eff_current[0, 3]
share_1 = eff_current[0, 1] / eff_out
share_3 = eff_current[0, 3] / eff_out
print('In the infinite-time, stationary case, a share of ' + \
      str(share_3) + ' outflow is via 3, while a share of '+ \
      str(share_1)+' outflow is via 1')


# TPT for the periodic case
# use as transition matrix T + wL, where w varies from 1..0..-1...0 
# L is a zero-rowsum matrix, T is a transition matrix
example_name = 'small_network_periodic'

M = 6  # 6 size of period

# transition matrix at time k
def P_p(k):
    # varies the transition matrices periodically, by weighting the added
    # matrix L with weights 1..0..-1.. over one period
    return T + np.cos(k*2.*np.pi/M)*L

# instantiate
small_periodic = periodic.tpt(
    P_p,
    M,
    ind_A,
    ind_B,
    ind_C,
)
# compute statistics
small_periodic.compute_statistics()
# save statistics
npz_path = os.path.join(data_path, example_name + '.npz')
small_periodic.save_statistics(npz_path)


# TPT for a finite time interval, time-homogeneous dynamics
example_name = 'small_network_finite'

# transition matrix at time n
def P_hom(n):
    return P

N = 5  # size of finite time interval

# initial density
init_dens_small_finite = small._stat_dens
# instantiate
small_finite = finite.tpt(
    P_hom,
    N,
    ind_A,
    ind_B,
    ind_C,
    init_dens_small_finite,
)
# compute statistics
small_finite.compute_statistics()
# save statistics
npz_path = os.path.join(data_path, example_name + '.npz')
small_finite.save_statistics(npz_path)


# TPT in finite time, time-inhomogeneous transition probabilities
example_name = 'small_network_inhom'
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
init_dens_small_inhom = small._stat_dens
# instantiate
small_inhom = finite.tpt(
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
npz_path = os.path.join(data_path, example_name + '.npz')
small_inhom.save_statistics(npz_path)


# TPT finite time extension to infinite time, convergence analysis
example_name = 'small_network_conv'
N_max = 150  # max value of N
q_f_conv = np.zeros((N_max, S))
q_b_conv = np.zeros((N_max, S))

# initial density
init_dens_small = small._stat_dens

for n in np.arange(1, N_max + 1):
    # extended time interval
    N_ex = n*2 + 1

    # instantiate
    small_finite_ex = finite.tpt(
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
npz_path = os.path.join(data_path, example_name + '.npz')
np.savez(
    npz_path,
    q_f=q_f_conv,
    q_b=q_b_conv,
)
