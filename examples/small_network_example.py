from pytpt import stationary  
from pytpt import periodic  
from pytpt import finite  
  
import functools
import numpy as np
 
import os.path
 
# define directories path to save the data and figures 
MY_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MY_PATH, 'data')

# load small network construction data
NETWORK_CONSTRUCTION = np.load(
    os.path.join(DATA_PATH, 'small_network_construction.npz'),
    allow_pickle=True,
)
STATES = NETWORK_CONSTRUCTION['states'].item()
T = NETWORK_CONSTRUCTION['T']
L = NETWORK_CONSTRUCTION['L']
K = NETWORK_CONSTRUCTION['K']
    
def P_p(k, M):
    ''' This method returns a transition matrix at time k over a period M
    '''
    # use as transition matrix T + wL, where w varies from 1..0..-1...0 
    # L is a zero-rowsum matrix, T is a transition matrix
    # varies the transition matrices periodically, by weighting the added
    # matrix L with weights 1..0..-1.. over one period
    return T + np.cos(k * 2. * np.pi / M) * L
    
def P_hom(n):
    ''' This method returns a time-homogeneous transition matrix.
    '''
    return T + L
    
def P_inhom(n):
    ''' This mthod returns a time-inhomogeneous transition matrix. 
    '''
    if np.mod(n, 2) == 0:
        return T + L + K
    else: 
        return T + L - K

def main():
    '''
    '''
    S = len(STATES)

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
    npz_path = os.path.join(DATA_PATH, example_name + '.npz')
    small.save_statistics(npz_path)

    #compute share along upper (1) and lower path (via 3)
    eff_current = small.eff_current
    eff_out = eff_current[0, 1] + eff_current[0, 3]
    share_1 = eff_current[0, 1] / eff_out
    share_3 = eff_current[0, 3] / eff_out
    print('In the infinite-time, stationary case, a share of ' + \
          str(share_3) + ' outflow is via 3, while a share of '+ \
          str(share_1)+' outflow is via 1')


    # TPT for the periodic case
    example_name = 'small_network_periodic'
    M = 6  # 6 size of period

    # instantiate
    small_periodic = periodic.tpt(
        functools.partial(P_p, M=M),
        M,
        ind_A,
        ind_B,
        ind_C,
    )
    # compute statistics
    small_periodic.compute_statistics()
    # save statistics
    npz_path = os.path.join(DATA_PATH, example_name + '.npz')
    small_periodic.save_statistics(npz_path)


    # TPT for a finite time interval, time-homogeneous dynamics
    example_name = 'small_network_finite'
    N = 5  # size of finite time interval
    # initial density
    init_dens_small_finite = small.stat_dens
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
    npz_path = os.path.join(DATA_PATH, example_name + '.npz')
    small_finite.save_statistics(npz_path)


    # TPT in finite time, time-inhomogeneous transition probabilities
    example_name = 'small_network_inhom'
    # size of time interval
    N_inhom = 5 

    # transition matrix at time n


    # initial density
    init_dens_small_inhom = small.stat_dens
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
    npz_path = os.path.join(DATA_PATH, example_name + '.npz')
    small_inhom.save_statistics(npz_path)


    # TPT finite time extension to infinite time, convergence analysis
    example_name = 'small_network_conv'
    N_max = 150  # max value of N
    q_f_conv = np.zeros((N_max, S))
    q_b_conv = np.zeros((N_max, S))

    # initial density
    init_dens_small = small.stat_dens

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
        [q_f_ex, q_b_ex] = small_finite_ex.committors()
        q_f_conv[n-1, :] = q_f_ex[n, :]
        q_b_conv[n-1, :] = q_b_ex[n, :]

    # save the transition statistics in npz files
    npz_path = os.path.join(DATA_PATH, example_name + '.npz')
    np.savez(
        npz_path,
        q_f=q_f_conv,
        q_b=q_b_conv,
    )

if __name__ == "__main__":
    main()
