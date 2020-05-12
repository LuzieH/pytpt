from validation import is_stochastic_matrix

import numpy as np
import pytest 
from pytpt import finite
import random
import functools

class TestFinite:
    @pytest.fixture
    def P(self, S, N):
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
        def P_N(k,N):
            gamma = k/(N-1) # ranges from 0 to 1 during time interval
            return (1-gamma)*P0 + gamma*P1

        return functools.partial(P_N, N=N) 
    
    @pytest.fixture
    def init_dens(self, S):
        '''  
        Args:
        S: int 
            dimension of the state space 
        '''
 
        init_dens = np.random.rand(S)
        init_dens = init_dens/np.sum(init_dens)

        return init_dens
    
    @pytest.fixture
    def states(self, S):
        ''' States classification
        '''
        states = np.empty(S, dtype='str') 

        # sorted list of two elements chosen from the set of integers 
        # between 0 and S-1 without replacement
        i, j = sorted(random.sample(range(0, S), 2))

        states[:i] = 'A'
        states[i:j] = 'B'
        states[j:] = 'C'
        
        return states

    @pytest.fixture
    def small_network_finite(self, states, P, init_dens, N):
        ''' initialize the tpt object 
        '''
       
        ind_A = np.where(states == 'A')[0]
        ind_B = np.where(states == 'B')[0]
        ind_C = np.where(states == 'C')[0]
        
        small_network_finite = finite.tpt(P, N, ind_A, ind_B, ind_C, init_dens)
        small_network_finite.committor()
        
        return small_network_finite

    def test_transition_matrix(self, small_network_finite):
        S = small_network_finite._S
        P = small_network_finite._P
        N = small_network_finite._N
        
        for n in range(N):
            assert P(n).shape == (S, S)
            assert np.isnan(P(n)).any() == False
            assert is_stochastic_matrix(P(n))

    def test_density(self, small_network_finite):
        S = small_network_finite._S
        density = small_network_finite.density()
        N = small_network_finite._N
        
        assert density.shape == (N,S)
        assert np.isnan(density).any() == False
        assert np.greater_equal(density.all(), 0) 
        assert np.less_equal(density.all(), 1) 
            

    def test_committors(self, small_network_finite):
        q_f, q_b = small_network_finite._q_f, small_network_finite._q_b
        S = small_network_finite._S
        N = small_network_finite._N     

        assert q_f.shape == (N,S)
        assert np.isnan(q_f).any() == False
        assert np.greater_equal(q_f.all(), 0) 
        assert np.less_equal(q_f.all(), 1) 

        assert q_b.shape == (N,S)
        assert np.isnan(q_b).any() == False
        assert np.greater_equal(q_b.all(), 0) 
        assert np.less_equal(q_b.all(), 1) 
        
    def test_reac_density(self, small_network_finite):
        reac_dens = small_network_finite.reac_density()
        norm_reac_dens = small_network_finite.norm_reac_density()
        S = small_network_finite._S
        N = small_network_finite._N  
        
        assert reac_dens.shape == (N,S)
        assert np.isnan(reac_dens).any() == False
        assert np.greater_equal(reac_dens.all(), 0) 
        assert np.less_equal(reac_dens.all(), 1) 
        
        assert norm_reac_dens.shape == (N,S)
        assert np.greater_equal(norm_reac_dens.all(), 0) 
        assert np.less_equal(norm_reac_dens.all(), 1)
        
 
    def test_current(self, small_network_finite):
        reac_current, eff_current = small_network_finite.reac_current()
        S = small_network_finite._S
        N = small_network_finite._N  
        
        assert reac_current.shape == (N,S,S)
        assert np.greater_equal(reac_current.all(), 0) 
        assert np.less_equal(reac_current.all(), 1) 
        
        assert eff_current.shape == (N,S,S)
        assert np.greater_equal(eff_current.all(), 0) 
        assert np.less_equal(eff_current.all(), 1)
 