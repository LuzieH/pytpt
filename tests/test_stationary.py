from pytpt.validation import is_stochastic_matrix, is_irreducible_matrix

import numpy as np
import pytest
from pytpt import stationary
import random

class TestStationary:
    @pytest.fixture(scope='class')
    def P_random(self, S, seed):
        ''' Random stationary transition matrix
        Args:
        S: int
            dimension of the state space
        seed: int
            seed
        '''
        # set seed
        np.random.seed(seed)

        # create random matrix uniformly distributed over [0, 1)
        P = np.random.rand(S, S)

        # normalize its values such that it is a stochastic matrix
        P = np.divide(P, np.sum(P, axis=1).reshape(S, 1))

        return P

    @pytest.fixture(scope='class')
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
    def P_small_network(self, shared_datadir):
        ''' Transition matrix of the small network example
        '''
        small_network_construction = np.load(
            shared_datadir / 'small_network_construction.npz',
            allow_pickle=True,
        )
        T = small_network_construction['T']
        L = small_network_construction['L']

        return T + L

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
    def tpt_stationary(self, states_random, P_random, states_small_network,
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
        
        tpt_stationary = stationary.tpt(P, ind_A, ind_B, ind_C)
        tpt_stationary.forward_committor()
        tpt_stationary.backward_committor()
        
        return tpt_stationary

    def test_transition_matrix(self, tpt_stationary):
        S = tpt_stationary.S
        P = tpt_stationary.P

        assert P.shape == (S, S)
        assert np.isnan(P).any() == False
        assert is_stochastic_matrix(P)
        assert is_irreducible_matrix(P)

    def test_backward_transition_matrix(self, tpt_stationary):
        S = tpt_stationary.S
        stationary_density = tpt_stationary.stationary_density()
        P = tpt_stationary.P
        P_back = tpt_stationary.backward_transitions()

        assert P_back.shape == (S, S)
        assert np.isnan(P_back).any() == False
        assert is_stochastic_matrix(P_back)

        for i in np.arange(S):
            for j in np.arange(S):
                assert np.isclose(
                    stationary_density[i] * P_back[i, j],
                    stationary_density[j] * P[j, i],
                )

    def test_stationary_density(self, tpt_stationary):
        S = tpt_stationary.S
        stationary_density = tpt_stationary.stationary_density()
        P = tpt_stationary.P
        P_back = tpt_stationary.backward_transitions()
        
        assert stationary_density.shape == (S,)
        assert np.isnan(stationary_density).any() == False
        assert np.greater_equal(stationary_density, 0).all() 
        assert np.less_equal(stationary_density, 1).all() 
        assert np.isclose(stationary_density.dot(P), stationary_density).all() 
        assert np.isclose(stationary_density.dot(P_back), stationary_density).all()

    def test_committors(self, tpt_stationary):
        q_f, q_b = tpt_stationary.q_f, tpt_stationary.q_b
        S = tpt_stationary.S

        assert q_f.shape == (S,)
        assert np.isnan(q_f).any() == False
        assert np.greater_equal(q_f, 0).all() 
        assert np.less_equal(q_f, 1).all() 

        assert q_b.shape == (S,)
        assert np.isnan(q_b).any() == False
        assert np.greater_equal(q_b, 0).all() 
        assert np.less_equal(q_b, 1).all() 

    def test_reac_density(self, tpt_stationary):
        reac_dens = tpt_stationary.reac_density()
        reac_norm_fact = tpt_stationary.reac_norm_factor()
        norm_reac_dens = tpt_stationary.norm_reac_density()
        S = tpt_stationary.S

        assert reac_dens.shape == (S,)
        assert np.isnan(reac_dens).any() == False
        assert np.greater_equal(reac_dens, 0).all() 
        assert np.less_equal(reac_dens, 1).all() 
        
        assert norm_reac_dens.shape == (S,)
        assert np.isnan(norm_reac_dens).any() == False
        assert np.greater_equal(norm_reac_dens, 0).all() 
        assert np.less_equal(norm_reac_dens, 1).all()
        
        assert np.isclose(reac_dens, reac_norm_fact * norm_reac_dens).all()
        
    def test_current(self, tpt_stationary):
        reac_current, eff_current = tpt_stationary.reac_current()
        S = tpt_stationary.S

        assert reac_current.shape == (S,S)
        assert np.isnan(reac_current).any() == False
        assert np.greater_equal(reac_current, 0).all() 
        assert np.less_equal(reac_current, 1).all() 
        
        assert eff_current.shape == (S,S)
        assert np.isnan(eff_current).any() == False
        assert np.greater_equal(eff_current, 0).all() 
        assert np.less_equal(eff_current, 1).all()

    def test_broadcasting_backward_transitions(self, tpt_stationary):
        # compute P_back without broadcasting
        S = tpt_stationary.S
        P = tpt_stationary.P
        stat_dens = tpt_stationary.stat_dens
        P_back = np.zeros(np.shape(P))
        for i in np.arange(S):
            for j in np.arange(S):
                if stat_dens[j] > 0:
                    P_back[j, i] = P[i, j] * stat_dens[i] / stat_dens[j]

        # compute P_back (with broadcasting)
        P_back_broadcast = tpt_stationary.backward_transitions()

        assert P_back_broadcast.shape == P_back.shape
        assert np.allclose(P_back_broadcast, P_back)


    def test_broadcasting_current(self, tpt_stationary):
        # compute current and effective current without broadcasting 
        S = tpt_stationary.S
        P = tpt_stationary.P
        stat_dens = tpt_stationary.stat_dens
        q_f = tpt_stationary.q_f
        q_b = tpt_stationary.q_b
        current = np.zeros(np.shape(P))
        eff_current = np.zeros(np.shape(P))
        for i in np.arange(S):
            for j in np.arange(S):
                current[i, j] = stat_dens[i] * q_b[i] * P[i, j] * q_f[j]
                if i + 1 > j:
                    eff_current[i, j] = np.max(
                        [0, current[i, j] - current[j, i]]
                    )
                    eff_current[j, i] = np.max(
                        [0, current[j, i] - current[i, j]]
                    )
        # compute current and effective current (with broadcasting)
        current_broadcast, eff_current_broadcast = tpt_stationary.reac_current()

        assert current_broadcast.shape == current.shape
        assert eff_current_broadcast.shape == eff_current.shape
        assert np.allclose(current_broadcast, current)
        assert np.allclose(eff_current_broadcast, eff_current)

