import numpy as np


class transitions_mcs:
    """Calculates committor probabilities and transition statistics of 
    Markov chain models"""

    def __init__(self, P,  ind_A, ind_B,  ind_C, stat_dens=None):
        """
        Initialize an instance by defining the transition matrix and the sets 
        between which the transition statistics should be computed.

        Parameters:
        P: array
            irreducible and row-stochastic (rows sum to 1) transition matrix  
            of size S x S, S is the size of the state space St = {1,2,...,S} 
        ind_A: array
            set of indices of the state space that belong to the set A
        ind_B: array
            set of indices of the state space that belong to the set B
        ind_C: array
            set of indices of the state space that belong to the transition 
            region C, i.e. the set C  =  St\(A u B)        
        stat_dens: array
            stationary distribution of P, normalized
            or if None, the density will be computed automatically
        """

        self._P = P
        self._stat_dens = stat_dens
        self._ind_A = ind_A
        self._ind_B = ind_B
        self._ind_C = ind_C
        self._S = np.shape(self._P)[0]  # size of state space

        self._P_back = None  # transition matrix of time-reversed process
        self._q_b = None  # backward committor
        self._q_f = None  # forward committor
        self._reac_dens = None  # reactive density
        self._current = None  # reactive current
        self._eff_current = None  # effective reactive current
        self._rate = None  # rate of transitions from A to B
        self._current_dens = None  # density of the effective current

        # compute the stationary density if its not given
        if self._stat_dens is None:
            self._stat_dens = self.stationary_density()

    def stationary_density(self):
        """
        Computes the stationary density of the transition matrix as the eigenvector
        of P with eigenvalue 1.
        """

        # compute stationary density
        eigv, eig = np.linalg.eig(np.transpose(self._P))
        # get index of eigenvector with eigenvalue 1 (up to small numerical error)
        index = np.where(np.isclose(eigv, 1))[0]
        # normalize
        stat_dens = (np.real(eig[:, index]) /
                     np.sum(np.real(eig[:, index]))).flatten()

        return stat_dens

    def committor(self):
        """
        Function that computes the forward committor q_f (probability that the 
        particle will next go to B rather than A) and backward commitor q_b 
        (probability that the system last came from A rather than B).
        """

        # compute backward transition matrix
        P_back = np.zeros(np.shape(self._P))
        for i in np.arange(self._S):
            for j in np.arange(self._S):
                P_back[j, i] = self._P[i, j] * \
                    self._stat_dens[i]/self._stat_dens[j]
        self._P_back = P_back

        # transition matrices from states in C to C
        P_C = self._P[np.ix_(self._ind_C, self._ind_C)]  # forward
        # backward in time
        P_back_C = self._P_back[np.ix_(self._ind_C, self._ind_C)]

        # amd from C to B
        P_CB = self._P[np.ix_(self._ind_C, self._ind_B)]
        P_back_CA = P_back[np.ix_(self._ind_C, self._ind_A)]

        q_f = np.zeros(self._S)
        # compute forward committor on C, the transition region
        b = np.sum(P_CB, axis=1)
        inv1 = np.linalg.inv(np.diag(np.ones(np.size(self._ind_C)))-P_C)
        q_f[np.ix_(self._ind_C)] = inv1.dot(b)
        # add entries to the forward committor vector on A, B
        # (i.e. q_f is 0 on A, 1 on B)
        q_f[np.ix_(self._ind_B)] = 1

        q_b = np.zeros(self._S)
        # compute backward committor on C
        a = np.sum(P_back_CA, axis=1)
        inv2 = np.linalg.inv(np.diag(np.ones(np.size(self._ind_C)))-P_back_C)
        q_b[np.ix_(self._ind_C)] = inv2.dot(a)
        # add entries to forward committor vector on A, B
        # (i.e. q_b is 1 on A, 0 on B)
        q_b[np.ix_(self._ind_A)] = 1

        self._q_b = q_b
        self._q_f = q_f

        return self._q_f, self._q_b

    def reac_density(self):
        """
        Given the forward and backward committor and the stationary density, 
        we can compute the normalized density of reactive trajectories, 
        i.e. the probability to be at x in St, given the chain is reactive.
        """
        assert self._q_f.all() != None, "The committor functions need \
        first to be computed by using the method committor"

        reac_dens = np.multiply(
            self._q_b, np.multiply(self._stat_dens, self._q_f))
        self._reac_dens = reac_dens/np.sum(reac_dens)  # normalization
        return self._reac_dens

    def reac_current(self):
        """
        Computes the reactive current current[i,j] between nodes i and j, as the 
        flow of reactive trajectories from i to j during one time step. 
        """
        assert self._q_f.all() != None, "The committor functions  need \
        first to be computed by using the method committor"

        current = np.zeros(np.shape(self._P))
        eff_current = np.zeros(np.shape(self._P))
        for i in np.arange(self._S):
            for j in np.arange(self._S):
                current[i, j] = self._stat_dens[i] * \
                    self._q_b[i]*self._P[i, j]*self._q_f[j]
                if i+1 > j:
                    eff_current[i, j] = np.max(
                        [0, current[i, j]-current[j, i]])
                    eff_current[j, i] = np.max(
                        [0, current[j, i]-current[i, j]])
        self._current = current
        self._eff_current = eff_current
        return self._current, self._eff_current

    def transition_rate(self):
        """
        The transition rate is the average flow of reactive trajectories out of A,
        which is the same as the average rate into B
        """

        assert self._current.all() != None, "The reactive current first needs \
        to be computed by using the method reac_current"

        self._rate = np.sum(self._current[self._ind_A, :])
        return self._rate

    def current_density(self):
        """
        The current density in a node is the sum of effective currents 
        over all neighbours of the node.
        """

        assert self._current.all() != None, "The reactive current first needs \
        to be computed by using the method reac_current"

        current_dens = np.zeros(self._S)
        for i in range(self._S):  # self._ind_C:
            current_dens[i] = np.sum(self._eff_current[i, :])
        self._current_dens = current_dens
        return self._current_dens


# todo: def compute_all(self):
# todo future: method to sample realizations -> get reactives densities thereof
# it's also a check of all the quantities
# todo: add to github repository
# todo: switch to python 3
