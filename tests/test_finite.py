from validation import is_stochastic_matrix, is_irreducible_matrix

import numpy as np
import pytest 
from pytpt import finite, stationary
import random
import functools

class TestFinite:
    @pytest.fixture
    def P_random(self, S, N):
        ''' Random finite-time transition matrix
        Args:
        S: int 
            dimension of the state space 
        '''
        # create random matrix uniformly distributed over [0, 1) and normalize
        # at time point mod 0
        P0 = np.random.rand(S, S)
        P0 = np.divide(P0, np.sum(P0, axis=1).reshape(S, 1))
        # at last time point 
        P1 = np.random.rand(S, S)
        P1 = np.divide(P1, np.sum(P1, axis=1).reshape(S, 1))
        # transition matrix interpolates between P0 and P1 during time interval
        def P_N(k, N):
            gamma = k / (N-1) # ranges from 0 to 1 during time interval
            return (1 - gamma) * P0 + gamma * P1
        
        return functools.partial(P_N, N=N) 
    
    @pytest.fixture
    def init_dens_random(self, S):
        '''  
        Args:
        S: int 
            dimension of the state space 
        '''
        init_dens = np.random.rand(S)
        init_dens = init_dens / np.sum(init_dens)
        return init_dens
    
    @pytest.fixture
    def states_random(self, S):
        ''' States classification
        '''
        states = np.empty(S, dtype='str') 
        # sorted list of two elements chosen from the set of integers 
        # between 0 and S-1 without replacement
        i, j = sorted(random.sample(range(1, S), 2))
        states[:i] = 'A'
        states[i:j] = 'B'
        states[j:] = 'C'
        return states
    
    @pytest.fixture
    def P_small_network(self, shared_datadir):
        ''' Finite-time transition matrix of the small network example
        '''
        small_network_construction = np.load(
            shared_datadir / 'small_network_construction.npz',
            allow_pickle=True, 
        )
        T = small_network_construction['T']
        L = small_network_construction['L']
        def P_hom(n):
            return T + L

        return P_hom
    
    @pytest.fixture
    def init_dens_small_network(self, shared_datadir):
        '''  
        '''
        small_network_construction = np.load(
            shared_datadir / 'small_network_construction.npz',
            allow_pickle=True, 
        )
        T = small_network_construction['T']
        L = small_network_construction['L']
        states = small_network_construction['states'].item()
        
        ind_A = np.array([key for key in states if states[key] == 'A'])
        ind_B = np.array([key for key in states if states[key] == 'B'])
        ind_C = np.array([key for key in states if states[key] == 'C'])

        tpt_stationary = stationary.tpt(T + L, ind_A, ind_B, ind_C)
        init_dens = tpt_stationary.stationary_density()
        
        return init_dens
    
    @pytest.fixture
    def states_small_network(self, shared_datadir):
        ''' States classification of the small network example
        '''
        small_network_construction = np.load(
            shared_datadir / 'small_network_construction.npz',
            allow_pickle=True, 
        )
        states = small_network_construction['states'].item()
        return states

    @pytest.fixture
    def tpt_finite(self, N, states_random, P_random, init_dens_random, 
                   states_small_network, P_small_network, init_dens_small_network, small_network):
        ''' initialize the tpt object 
        '''
        if small_network:
            states = states_small_network
            P = P_small_network
            init_dens = init_dens_small_network
            ind_A = np.array([key for key in states if states[key] == 'A'])
            ind_B = np.array([key for key in states if states[key] == 'B'])
            ind_C = np.array([key for key in states if states[key] == 'C'])
        else:
            states = states_random
            P = P_random
            init_dens = init_dens_random
            ind_A = np.where(states == 'A')[0]
            ind_B = np.where(states == 'B')[0]
            ind_C = np.where(states == 'C')[0]
        
        tpt_finite = finite.tpt(P, N, ind_A, ind_B, ind_C, init_dens)
        tpt_finite.committor()
        
        return tpt_finite

    def test_transition_matrix(self, tpt_finite):
        S = tpt_finite._S
        P = tpt_finite._P
        N = tpt_finite._N
        
        for n in range(N):
            assert P(n).shape == (S, S)
            assert np.isnan(P(n)).any() == False
            assert is_stochastic_matrix(P(n))
            assert is_irreducible_matrix(P(n))

    def test_density(self, tpt_finite):
        S = tpt_finite._S
        density = tpt_finite.density()
        N = tpt_finite._N
        P = tpt_finite._P
        
        assert density.shape == (N, S)
        assert np.isnan(density).any() == False
        assert np.greater_equal(density.all(), 0) 
        assert np.less_equal(density.all(), 1) 
        
        for n in range(N - 1):
            assert np.isclose(density[n, :].dot(P(n)), density[n + 1, :]).all()
            

    def test_committors(self, tpt_finite):
        q_f, q_b = tpt_finite._q_f, tpt_finite._q_b
        S = tpt_finite._S
        N = tpt_finite._N     

        assert q_f.shape == (N,S)
        assert np.isnan(q_f).any() == False
        assert np.greater_equal(q_f, 0).all() 
        assert np.less_equal(q_f, 1).all() 

        assert q_b.shape == (N,S)
        assert np.isnan(q_b).any() == False
        assert np.greater_equal(q_b, 0).all() 
        assert np.less_equal(q_b, 1).all() 
        
    def test_reac_density(self, tpt_finite):
        reac_dens = tpt_finite.reac_density()
        norm_reac_dens = tpt_finite.norm_reac_density()
        S = tpt_finite._S
        N = tpt_finite._N  
        
        assert reac_dens.shape == (N, S)
        assert np.isnan(reac_dens).any() == False
        assert (np.fmin(reac_dens,0)>=0).all() #np.greater_equal(reac_dens, 0).all() 
        assert (np.fmin(reac_dens,1)<=1).all() #np.less_equal(reac_dens, 1).all() 
        
        assert norm_reac_dens.shape == (N, S)
        assert (np.fmin(norm_reac_dens,0)>=0).all() #np.greater_equal(norm_reac_dens, 0).all() 
        assert (np.fmin(norm_reac_dens,1)<=1).all() #np.less_equal(norm_reac_dens, 1).all()
        
 
    def test_current(self, tpt_finite):
        [reac_current, eff_current] = tpt_finite.reac_current()
        S = tpt_finite._S
        N = tpt_finite._N  
        
        assert reac_current.shape == (N, S, S)
        assert (np.fmin(reac_current,0)>=0).all() #np.greater_equal(reac_current, 0).all() 
        assert (np.fmin(reac_current,1)<=1).all() #np.less_equal(reac_current, 1).all() 
        
        assert eff_current.shape == (N, S, S)
        assert (np.fmin(eff_current,0)>=0).all() #np.greater_equal(eff_current, 0).all() 
        assert (np.fmin(eff_current,1)<=1).all() #np.less_equal(eff_current, 1).all()
