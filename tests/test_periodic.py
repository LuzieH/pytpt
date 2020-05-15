from validation import is_stochastic_matrix

import numpy as np
import pytest 
from pytpt import periodic
import random
import functools

class TestPeriodic:
    @pytest.fixture
    def P(self, S, M):
        ''' Random periodic stationary transition matrix
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
        
        # transition matrix interpolates between P0 and P1 during period
        def P_M(k, M):
            gamma = np.mod(k, M)/(M-1) # ranges from 0 to 1 during each period
            return (1-gamma)*P0 + gamma*P1

        return functools.partial(P_M, M=M) 

    @pytest.fixture
    def states(self, S):
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
    def small_network_periodic(self, states, P, M):
        ''' initialize the tpt object 
        '''
 
        ind_A = np.where(states == 'A')[0]
        ind_B = np.where(states == 'B')[0]
        ind_C = np.where(states == 'C')[0]
        
        small_network_periodic = periodic.tpt(P, M, ind_A, ind_B, ind_C)
        small_network_periodic.committor()
        
        return small_network_periodic

    def test_transition_matrix(self, small_network_periodic):
        S = small_network_periodic._S
        P = small_network_periodic._P
        M = small_network_periodic._M
        
        for m in range(M):
            assert P(m).shape == (S, S)
            assert np.isclose(P(m), P(M+m)).all()
            assert np.isnan(P(m)).any() == False
            assert is_stochastic_matrix(P(m))

    def test_stationary_density(self, small_network_periodic):
        S = small_network_periodic._S
        stationary_density = small_network_periodic.stationary_density()
        M = small_network_periodic._M
        
        assert stationary_density.shape == (M,S)
        assert np.isnan(stationary_density).any() == False
        assert np.greater_equal(stationary_density.all(), 0) 
        assert np.less_equal(stationary_density.all(), 1) 
            
    def test_backward_transition_matrix(self, small_network_periodic):
        S = small_network_periodic._S
        stationary_density = small_network_periodic.stationary_density()
        P = small_network_periodic._P
        P_back = small_network_periodic.backward_transitions()
        M = small_network_periodic._M        
        
        for m in range(M):
            assert P_back(m).shape == (S, S)
            assert np.isnan(P_back(m)).any() == False
            assert is_stochastic_matrix(P_back(m))

            for i in np.arange(S):
                for j in np.arange(S):
                    assert np.isclose(
                        stationary_density[m,i] * P_back(m)[i, j],
                        stationary_density[m-1,j] * P(m-1)[j, i],
                    )

    def test_committors(self, small_network_periodic):
        q_f, q_b = small_network_periodic._q_f, small_network_periodic._q_b
        S = small_network_periodic._S
        M = small_network_periodic._M     

        assert q_f.shape == (M,S)
        assert np.isnan(q_f).any() == False
        assert np.greater_equal(q_f.all(), 0) 
        assert np.less_equal(q_f.all(), 1) 

        assert q_b.shape == (M,S)
        assert np.isnan(q_b).any() == False
        assert np.greater_equal(q_b.all(), 0) 
        assert np.less_equal(q_b.all(), 1) 
        
    def test_reac_density(self, small_network_periodic):
        reac_dens = small_network_periodic.reac_density()
        reac_norm_factor = small_network_periodic.reac_norm_factor()
        norm_reac_dens = small_network_periodic.norm_reac_density()
        S = small_network_periodic._S
        M = small_network_periodic._M  
        
        assert reac_dens.shape == (M,S)
        assert np.isnan(reac_dens).any() == False
        assert np.greater_equal(reac_dens.all(), 0) 
        assert np.less_equal(reac_dens.all(), 1) 
        
        assert norm_reac_dens.shape == (M,S)
        assert np.greater_equal(norm_reac_dens.all(), 0) 
        assert np.less_equal(norm_reac_dens.all(), 1)
        
   
    def test_current(self, small_network_periodic):
        reac_current, eff_current = small_network_periodic.reac_current()
        S = small_network_periodic._S
        M = small_network_periodic._M  
        
        assert reac_current.shape == (M,S,S)
        assert np.isnan(reac_current).any() == False
        assert np.greater_equal(reac_current.all(), 0) 
        assert np.less_equal(reac_current.all(), 1) 
        
        assert eff_current.shape == (M,S,S)
        assert np.isnan(eff_current).any() == False
        assert np.greater_equal(eff_current.all(), 0) 
        assert np.less_equal(eff_current.all(), 1)
 
