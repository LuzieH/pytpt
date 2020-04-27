from validation import is_stochastic_matrix

import numpy as np
import pytest 
from pytpt import stationary  

class TestStationary:
    @pytest.fixture
    def P(self):
        ''' Stationary transition matrix
        '''
        P = np.zeros((3, 3))
        P[0, 0] = 0.7
        P[0, 1] = 0.2
        P[0, 2] = 0.1
        P[1, 0] = 0.5
        P[1, 1] = 0
        P[1, 2] = 0.5
        P[2, 0] = 0.1
        P[2, 1] = 0.2
        P[2, 2] = 0.7

        return P
    @pytest.fixture
    def states(self):
        ''' States classification
        '''
        states = {
            0: 'A',
            1: 'C',
            2: 'B',
        }
        return states

    @pytest.fixture
    def small_network(self, states, P):
        ''' initialize the tpt object 
        '''
        ind_A = np.array([key for key in states if states[key] == 'A'])
        ind_B = np.array([key for key in states if states[key] == 'B'])
        ind_C = np.array([key for key in states if states[key] == 'C'])
        return stationary.tpt(P, ind_A, ind_B, ind_C)

    def test_transition_matrix(self, small_network):
        P = small_network._P
        S = small_network._S

        assert P.shape == (S, S)
        assert np.isnan(P).any() == False
        assert is_stochastic_matrix(P)

    def test_backward_transition_matrix(self, small_network):
        P_back = small_network.backward_transitions()
        S = small_network._S

        assert P_back.shape == (S, S)
        assert np.isnan(P_back).any() == False
        assert is_stochastic_matrix(P_back)

    def test_stationary_density(self, small_network):
        stationary_density = small_network.stationary_density()
        S = small_network._S
        P = small_network._P
        P_back = small_network.backward_transitions()
        
        assert stationary_density.shape == (S,)
        assert np.isnan(stationary_density).any() == False
        assert np.greater_equal(stationary_density.all(), 0) 
        assert np.less_equal(stationary_density.all(), 1) 
        assert np.isclose(stationary_density.dot(P),stationary_density).all() 
        assert np.isclose(stationary_density.dot(P_back),stationary_density).all()  

    def test_committors(self, small_network):
        q_f, q_b = small_network.committor()
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
        
        assert np.isclose(reac_dens, reac_norm_factor*norm_reac_dens)
