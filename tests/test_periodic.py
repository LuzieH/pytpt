from pytpt.validation import is_stochastic_matrix, is_irreducible_matrix

import numpy as np
import pytest 
from pytpt import periodic
import random
import functools

class TestPeriodic:
    @pytest.fixture
    def P_random(self, S, M, seed):
        ''' Random periodic stationary transition matrix
        Args:
        S: int 
            dimension of the state space 
        seed: int
            seed
        '''
        # set seed
        np.random.seed(seed)

        # create random matrix uniformly distributed over [0, 1) and normalize
        # at time point mod 0
        P0 = np.random.rand(S, S)
        P0 = np.divide(P0, np.sum(P0, axis=1).reshape(S, 1))
        # at last time point 
        P1 = np.random.rand(S, S)
        P1 = np.divide(P1, np.sum(P1, axis=1).reshape(S, 1))
        
        # transition matrix interpolates between P0 and P1 during period
        def P_M(k, M):
            gamma = np.mod(k, M) / (M-1) # ranges from 0 to 1 during each period
            return (1-gamma)*P0 + gamma*P1

        return functools.partial(P_M, M=M) 

    @pytest.fixture
    def states_random(self, S, seed):
        ''' States classification
        Args:
        S: int
            dimension of the state space
        seed: int
            seed
        '''
        # set seed
        random.seed(seed)

        states = np.empty(S, dtype='str') 

        # sorted list of two elements chosen from the set of integers 
        # between 0 and S-1 without replacement
        i, j = sorted(random.sample(range(1, S), 2))

        states[:i] = 'A'
        states[i:j] = 'B'
        states[j:] = 'C'
        
        return states
    
    @pytest.fixture
    def P_small_network(self, shared_datadir, M):
        ''' Periodic transition matrix of the small network example
        '''
        small_network_construction = np.load(
            shared_datadir / 'small_network_construction.npz',
            allow_pickle=True, 
        )
        T = small_network_construction['T']
        L = small_network_construction['L']

        def P_p(k, M):
            return T + np.cos(k * 2. * np.pi / M) * L
        
        return functools.partial(P_p, M=M)
    
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
    def tpt_periodic(self, M, states_random, P_random, states_small_network,
                     P_small_network, small_network):
        ''' initialize the tpt object 
        '''
        if small_network:
            states = states_small_network
            P = P_small_network
            ind_A = np.array([key for key in states if states[key] == 'A'])
            ind_B = np.array([key for key in states if states[key] == 'B'])
            ind_C = np.array([key for key in states if states[key] == 'C'])            
            
        else:
            states = states_random
            P = P_random
            ind_A = np.where(states == 'A')[0]
            ind_B = np.where(states == 'B')[0]
            ind_C = np.where(states == 'C')[0]
        
        tpt_periodic = periodic.tpt(P, M, ind_A, ind_B, ind_C)
        tpt_periodic.forward_committor()
        tpt_periodic.backward_committor()
        
        return tpt_periodic

    def test_transition_matrix(self, tpt_periodic):
        S = tpt_periodic.S
        M = tpt_periodic.M
        P = tpt_periodic.P
        
        for m in range(M):
            assert P(m).shape == (S, S)
            assert np.isclose(P(m), P(M + m)).all()
            assert np.isnan(P(m)).any() == False
            assert is_stochastic_matrix(P(m))
            assert is_irreducible_matrix(P(m))

    def test_stationary_density(self, tpt_periodic):
        S = tpt_periodic.S
        M = tpt_periodic.M
        stationary_density = tpt_periodic.stationary_density()
        
        assert stationary_density.shape == (M, S)
        assert np.isnan(stationary_density).any() == False
        assert np.greater_equal(stationary_density, 0).all() 
        assert np.less_equal(stationary_density, 1).all() 
            
    def test_backward_transition_matrix(self, tpt_periodic):
        S = tpt_periodic.S
        M = tpt_periodic.M        
        stationary_density = tpt_periodic.stationary_density()
        P = tpt_periodic.P
        P_back = tpt_periodic.backward_transitions()
        
        for m in range(M):
            assert P_back(m).shape == (S, S)
            assert np.isnan(P_back(m)).any() == False
            assert is_stochastic_matrix(P_back(m))

            for i in np.arange(S):
                for j in np.arange(S):
                    assert np.isclose(
                        stationary_density[m, i] * P_back(m)[i, j],
                        stationary_density[m-1, j] * P(m-1)[j, i],
                    )

    def test_committors(self, tpt_periodic):
        S = tpt_periodic.S
        M = tpt_periodic.M     
        q_f, q_b = tpt_periodic.q_f, tpt_periodic.q_b

        assert q_f.shape == (M, S)
        assert np.isnan(q_f).any() == False
        assert np.greater_equal(q_f, 0).all() 
        assert np.less_equal(q_f, 1).all() 

        assert q_b.shape == (M, S)
        assert np.isnan(q_b).any() == False
        assert np.greater_equal(q_b, 0).all() 
        assert np.less_equal(q_b, 1).all() 
        
    def test_reac_density(self, tpt_periodic):
        S = tpt_periodic.S
        M = tpt_periodic.M  
        reac_dens = tpt_periodic.reac_density()
        reac_norm_fact = tpt_periodic.reac_norm_factor()
        norm_reac_dens = tpt_periodic.norm_reac_density()
        
        assert reac_dens.shape == (M, S)
        assert np.isnan(reac_dens).any() == False
        assert np.greater_equal(reac_dens, 0).all() 
        assert np.less_equal(reac_dens, 1).all() 
        
        assert norm_reac_dens.shape == (M, S)
        assert np.greater_equal(norm_reac_dens, 0).all() 
        assert np.less_equal(norm_reac_dens, 1).all()
        
   
    def test_current(self, tpt_periodic):
        reac_current, eff_current = tpt_periodic.reac_current()
        S = tpt_periodic.S
        M = tpt_periodic.M  
        
        assert reac_current.shape == (M, S, S)
        assert np.isnan(reac_current).any() == False
        assert np.greater_equal(reac_current, 0).all() 
        assert np.less_equal(reac_current, 1).all() 
        
        assert eff_current.shape == (M, S, S)
        assert np.isnan(eff_current).any() == False
        assert np.greater_equal(eff_current, 0).all() 
        assert np.less_equal(eff_current, 1).all()
    
    def test_broadcasting_backward_transitions(self, tpt_periodic):
        # compute P_back without broadcasting
        S = tpt_periodic.S
        M = tpt_periodic.M
        P = tpt_periodic.P
        stat_dens = tpt_periodic.stat_dens
        
        P_back_m = np.zeros((M, S, S))
        for m in range(M):
            for i in np.arange(S):
                for j in np.arange(S):
                    if stat_dens[np.mod(m, M), j] > 0:
                        P_back_m[m, j, i] = P(m - 1)[i, j] \
                                          * stat_dens[np.mod(m - 1, M), i] \
                                          / stat_dens[np.mod(m, M), j]
        def P_back(k):
            return P_back_m[np.mod(k, M), :, :]


        # compute P_back (with broadcasting)
        P_back_broadcast = tpt_periodic.backward_transitions()
        
        for m in range(M):
            assert P_back_broadcast(m).shape == P_back(m).shape
            assert np.allclose(P_back_broadcast(m), P_back(m))


    def test_broadcasting_current(self, tpt_periodic):
        # compute current and effective current without broadcasting 
        S = tpt_periodic.S
        M = tpt_periodic.M
        P = tpt_periodic.P
        stat_dens = tpt_periodic.stat_dens
        q_f = tpt_periodic.q_f
        q_b = tpt_periodic.q_b

        current = np.zeros((M, S, S))
        eff_current = np.zeros((M, S, S))
        
        for m in range(M):
            for i in np.arange(S):
                for j in np.arange(S):
                    current[m, i, j] = stat_dens[m, i] \
                                     * q_b[m, i] \
                                     * P(m)[i, j] \
                                     * q_f[np.mod(m + 1, M), j]
                    if i + 1 > j:
                        eff_current[m, i, j] = np.maximum(
                            0, current[m, i, j] - current[m, j, i]
                        )
                        eff_current[m, j, i] = np.maximum(
                            0, current[m, j, i] - current[m, i, j]
                        )

        # compute current and effective current (with broadcasting)
        current_broadcast, eff_current_broadcast = tpt_periodic.reac_current()

        assert current_broadcast.shape == current.shape
        assert eff_current_broadcast.shape == eff_current.shape
        assert np.allclose(current_broadcast, current)
        assert np.allclose(eff_current_broadcast, eff_current)
