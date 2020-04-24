from validation import is_stochastic_matrix

import numpy as np
import pytest 

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
    

    def test_is_P_stochastic_matrix(self, P):
        assert is_stochastic_matrix(P)
