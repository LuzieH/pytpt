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

    def test_transition_matrix(self, P):
        assert is_stochastic_matrix(P)

    def test_stationary_density(self, small_network):
        stationary_density = small_network.stationary_density()
        assert np.greater_equal(stationary_density.all(), 0) 
        assert np.less_equal(stationary_density.all(), 1) 

    def test_committors(self, small_network):
        q_f, q_b = small_network.committor()
        assert np.greater_equal(q_f.all(), 0) 
        assert np.less_equal(q_f.all(), 1) 
        assert np.greater_equal(q_b.all(), 0) 
        assert np.less_equal(q_b.all(), 1) 
