from validation import is_stochastic_matrix, is_irreducible_matrix

import numpy as np
import pytest 
from pytpt import stationary  
import random

class TestStationary:
    @pytest.fixture
    def P(self, S):
        ''' Random stationary transition matrix
        Args:
        S: int 
            dimension of the state space 
        '''
        # create random matrix uniformly distributed over [0, 1)
        P = np.random.rand(S, S)

        # normalize its values such that it is a stochastic matrix
        P = np.divide(P, np.sum(P, axis=1).reshape(S, 1))

        return P

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
    def small_network(self, states, P):
        ''' initialize the tpt object 
        '''
        ind_A = np.where(states == 'A')[0]
        ind_B = np.where(states == 'B')[0]
        ind_C = np.where(states == 'C')[0]
        
        small_network = stationary.tpt(P, ind_A, ind_B, ind_C)
        small_network.committor()
        
        return small_network

    def test_transition_matrix(self, small_network):
        S = small_network._S
        P = small_network._P

        assert P.shape == (S, S)
        assert np.isnan(P).any() == False
        assert is_stochastic_matrix(P)
        assert is_irreducible_matrix(P)

    def test_backward_transition_matrix(self, small_network):
        S = small_network._S
        stationary_density = small_network.stationary_density()
        P = small_network._P
        P_back = small_network.backward_transitions()

        assert P_back.shape == (S, S)
        assert np.isnan(P_back).any() == False
        assert is_stochastic_matrix(P_back)

        for i in np.arange(S):
            for j in np.arange(S):
                assert np.isclose(
                    stationary_density[i] * P_back[i, j],
                    stationary_density[j] * P[j, i],
                )

    def test_stationary_density(self, small_network):
        S = small_network._S
        stationary_density = small_network.stationary_density()
        P = small_network._P
        P_back = small_network.backward_transitions()
        
        assert stationary_density.shape == (S,)
        assert np.isnan(stationary_density).any() == False
        assert np.greater_equal(stationary_density.all(), 0) 
        assert np.less_equal(stationary_density.all(), 1) 
        assert np.isclose(stationary_density.dot(P), stationary_density).all() 
        assert np.isclose(stationary_density.dot(P_back), stationary_density).all()

    def test_committors(self, small_network):
        q_f, q_b = small_network._q_f, small_network._q_b
        S = small_network._S

        assert q_f.shape == (S,)
        assert np.isnan(q_f).any() == False
        assert np.greater_equal(q_f.all(), 0) 
        assert np.less_equal(q_f.all(), 1) 

        assert q_b.shape == (S,)
        assert np.isnan(q_b).any() == False
        assert np.greater_equal(q_b.all(), 0) 
        assert np.less_equal(q_b.all(), 1) 

    def test_reac_density(self, small_network):
        reac_dens = small_network.reac_density()
        reac_norm_factor = small_network.reac_norm_factor()
        norm_reac_dens = small_network.norm_reac_density()
        S = small_network._S

        assert reac_dens.shape == (S,)
        assert np.isnan(reac_dens).any() == False
        assert np.greater_equal(reac_dens.all(), 0) 
        assert np.less_equal(reac_dens.all(), 1) 
        
        assert norm_reac_dens.shape == (S,)
        assert np.isnan(norm_reac_dens).any() == False
        assert np.greater_equal(norm_reac_dens.all(), 0) 
        assert np.less_equal(norm_reac_dens.all(), 1)
        
        assert np.isclose(reac_dens, reac_norm_factor * norm_reac_dens).all()
        
    def test_current(self, small_network):
        reac_current, eff_current = small_network.reac_current()
        S = small_network._S

        assert reac_current.shape == (S,S)
        assert np.isnan(reac_current).any() == False
        assert np.greater_equal(reac_current.all(), 0) 
        assert np.less_equal(reac_current.all(), 1) 
        
        assert eff_current.shape == (S,S)
        assert np.isnan(eff_current).any() == False
        assert np.greater_equal(eff_current.all(), 0) 
        assert np.less_equal(eff_current.all(), 1)
