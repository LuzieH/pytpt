from validation import is_stochastic_matrix

import numpy as np
import pytest 
from pytpt import periodic

class TestPeriodic:
    @pytest.fixture
    def P(self, S):
        ''' Random periodic stationary transition matrix
        Args:
        S: int 
            dimension of the state space 
        '''
        M = 2
        
        # create random matrix uniformly distributed over [0, 1) and normalize
        # at time point mod 0
        P0 = np.random.rand(S, S)
        P0 = np.divide(P0, np.sum(P0, axis=1).reshape(S, 1))
        # at time point mod 1
        P1 = np.random.rand(S, S)
        P1 = np.divide(P1, np.sum(P1, axis=1).reshape(S, 1))
        
        P = lambda k : P0 if np.mod(k,M)==0 else P1

        return P 

    @pytest.fixture
    def states(self, S):
        ''' States classification
        '''
        states = {
            0: 'A',
            1: 'B'
        }
        # remaining states are assigned to C
        states.update({i : 'C' for i in range(2,S)})
        
        return states

    @pytest.fixture
    def small_network_periodic(self, states, P):
        ''' initialize the tpt object 
        '''
        M = 2
        
        ind_A = np.array([key for key in states if states[key] == 'A'])
        ind_B = np.array([key for key in states if states[key] == 'B'])
        ind_C = np.array([key for key in states if states[key] == 'C'])
        
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
        
        for m in range(M):
            assert stationary_density[m,:].shape == (S,)
            assert np.isnan(stationary_density[m,:]).any() == False
            assert np.greater_equal(stationary_density[m,:].all(), 0) 
            assert np.less_equal(stationary_density[m,:].all(), 1) 
            
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
 