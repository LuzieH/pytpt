import numpy as np


class transitions_periodic:
    """Calculates committor probabilities and transition statistics of 
    Markov chain models with periodic forcing"""

    def __init__(self, P, M, ind_A, ind_B,  ind_C):
        """
        Initialize an instance by defining the periodically forced transition matrix
        and the sets
        between which the transition statistics should be computed.

        Parameters:
        P: function mapping time modulo (0,1,...,M-1) to the corresponding transition matirx
           (row-stochastic (rows sum to 1) transition matrix  
            of size S x S, S is the size of the state space St = {1,2,...,S})
           moreover the product of transition matrices should be irreducible
        M: int
            size of the period
        ind_A: array
            set of indices of the state space that belong to the set A
        ind_B: array
            set of indices of the state space that belong to the set B
        ind_C: array
            set of indices of the state space that belong to the transition 
            region C, i.e. the set C  =  St\(A u B)        
        """

        assert np.isclose(P(0), P(M)).all(), "The transition matrix function \
        needs to the time modulo M to the corresponding transition matrix."

        self._P = P
        self._M = M
        self._ind_A = ind_A
        self._ind_B = ind_B
        self._ind_C = ind_C
        self._S = np.shape(self._P(0))[0]  # size of the state space

        # compute the stationary density
        self._stat_dens = self.stationary_density()

        # compute the backward transitin matrix and store as function
        self._P_back = self.backward_transitions()

        self._q_b = None  # backward committor
        self._q_f = None  # forward committor
        self._reac_dens = None  # reactive density
        self._current = None  # reactive current
        self._eff_current = None  # effective reactive current
        self._rate = None  # rate of transitions from A to B
        self._current_dens = None  # density of the effective current

    def stationary_density(self):
        """
        Computes the periodically varying stationary densities of 
        the transition matrix and returns them.
        """

        stat_dens = np.zeros((self._M, self._S))

        # product of transition matrices over 1 period starting at 0
        P_bar = self._P(0)
        for m in np.arange(1, self._M):
            P_bar = P_bar.dot(self._P(m))

        # compute stationary density of P_bar
        eigv, eig = np.linalg.eig(np.transpose(P_bar))
        # get index of eigenvector with eigenvalue 1
        index = np.where(np.isclose(eigv, 1))[0]
        # normalize
        stat_dens[0, :] = (np.real(eig[:, index]) /
                           np.sum(np.real(eig[:, index]))).flatten()

        # compute remaining densities
        for m in np.arange(1, self._M):
            stat_dens[m, :] = stat_dens[m-1, :].dot(self._P(m-1))

        return stat_dens

    def backward_transitions(self):
        """
        Computes the transition matrix backwards in time. Returns a function 
        that for each time assigs the correct backward transition matrix modulo M.
        """
        P_back_m = np.zeros((self._M, self._S, self._S))

        for m in range(self._M):
            # compute backward transition matrix
            for i in np.arange(self._S):
                for j in np.arange(self._S):
                    P_back_m[m, j, i] = self._P(m-1)[i, j] *\
                        self._stat_dens[np.mod(
                            m-1, self._M), i]/self._stat_dens[np.mod(m, self._M), j]

        # store backward matrix in a function that assigns each time point to the
        # corresponding transition matrix
        def P_back(k):
            return P_back_m[np.mod(k, self._M), :, :]

        return P_back

    def committor(self):
        """
        Function that computes the forward committor q_f (probability that the 
        particle will next go to B rather than A) and backward commitor q_b 
        (probability that the system last came from A rather than B) of the periodic 
        system by using the stacked equations.
        """

        # dimension of sets A, B, C
        dim_A = np.size(self._ind_A)
        dim_B = np.size(self._ind_B)
        dim_C = np.size(self._ind_C)

        # forward committors q^+_0 at time 0
        # to solve: (I-D)q^+_0 = b

        # multiplied transition matrix over period with only transitions in C
        D = np.diag(np.ones(dim_C))
        b = np.zeros(dim_C)  # remaining part of the equation

        # filling D and b
        for m in np.arange(self._M):

            b = b + \
                (D.dot(self._P(m)[np.ix_(self._ind_C, self._ind_B)])).dot(
                    np.ones(dim_B))
            D = D.dot(self._P(m)[np.ix_(self._ind_C, self._ind_C)])

        # B = I-D
        B = np.diag(np.ones(dim_C)) - D

        # invert B
        inv_B = np.linalg.inv(B)

        # find q_0^+ on C
        q_f = np.zeros((self._M, self._S))
        q_f[0, self._ind_C] = inv_B.dot(b)
        # on B: q^+ = 1
        q_f[:, self._ind_B] = 1

        #q_f[self._M-1,self._ind_C] = self._P(self._M-1)[self._ind_C,:].dot(q_f[0,:])

        # compute committors at remaining times
        for m in np.arange(1, self._M)[::-1]:
            q_f[m, self._ind_C] = self._P(m)[self._ind_C, :].dot(
                q_f[np.mod(m+1, self._M), :])

        self._q_f = q_f

        # backward committor q^-_0 at time 0
        # to solve (I-D_back)q^-_0 = a

        # multiplied bakward transition matrix over all times with only transitions in C
        D_back = np.diag(np.ones(dim_C))
        a = np.zeros(dim_C)  # remaining part of the equation

        times = np.arange(1, self._M+1)[::-1]
        times[0] = 0

        for m in times:

            a = a + (D_back.dot(self._P_back(m)
                                [np.ix_(self._ind_C, self._ind_A)])).dot(np.ones(dim_A))
            D_back = D_back.dot(self._P_back(
                m)[np.ix_(self._ind_C, self._ind_C)])

        # A = I-D_back
        A = np.diag(np.ones(dim_C)) - D_back
        # invert A
        inv_A = np.linalg.inv(A)

        # find q_0^- on C
        q_b = np.zeros((self._M, self._S))
        q_b[0, self._ind_C] = inv_A.dot(a)
        q_b[:, self._ind_A] = 1

        # compute committor for remaining times
        for m in np.arange(1, self._M):
            q_b[m, self._ind_C] = self._P_back(
                m)[self._ind_C, :].dot(q_b[m-1, :])

        self._q_b = q_b

        return self._q_f, self._q_b

    def reac_density(self):
        """
        Given the forward and backward committor and the density, 
        we can compute the normalized density of reactive trajectories, 
        i.e. the probability to be at x in S at time m, given the chain is reactive.
        The function returns an array of the reactive density for each time 
        (with time as the first index of the array).
        """
        assert self._q_f.all() != None, "The committor functions need \
        first to be computed by using the method committor"

        reac_dens = np.zeros((self._M, self._S))

        for m in range(self._M):

            reac_dens[m, :] = np.multiply(self._q_b[m, :], np.multiply(
                self._stat_dens[m, :], self._q_f[m, :]))
            reac_dens[m, :] = reac_dens[m, :] / \
                np.sum(reac_dens[m, :])  # normalization

        self._reac_dens = reac_dens

        return self._reac_dens

    def reac_current(self):
        """
        Computes the reactive current current[i,j] between nodes i and j, as the 
        flow of reactive trajectories from i to j during one time step. 
        """
        assert self._q_f.all() != None, "The committor functions  need \
        first to be computed by using the method committor"

        current = np.zeros((self._M, self._S, self._S))
        eff_current = np.zeros((self._M, self._S, self._S))

        for m in range(self._M):

            for i in np.arange(self._S):
                for j in np.arange(self._S):
                    current[m, i, j] = self._stat_dens[m, i] * \
                        self._q_b[m, i]*self._P(m)[i, j]*self._q_f[m, j]

                    if i+1 > j:
                        eff_current[m, i, j] = np.max(
                            [0, current[m, i, j]-current[m, j, i]])
                        eff_current[m, j, i] = np.max(
                            [0, current[m, j, i]-current[m, i, j]])

        self._current = current
        self._eff_current = eff_current

        return self._current, self._eff_current

    def transition_rate(self):
        """
        The transition rate is the average flow of reactive trajectories out of 
        A at time m (first row) or into B at time m (second row)
        """

        assert self._current.all() != None, "The reactive current first needs \
        to be computed by using the method reac_current"

        # for each time m, sum of all currents out of A into S
        self._rate = np.zeros((2, self._M))
        self._rate[0, :] = np.sum(
            self._current[:, self._ind_A, :], axis=(1, 2))
        self._rate[1, :] = np.sum(
            self._current[:, :, self._ind_B], axis=(1, 2))

        return self._rate

    def current_density(self):
        """
        The current density in a node is the sum of effective currents 
        over all neighbours of the node.
        """

        assert self._current.all() != None, "The reactive current first needs \
        to be computed by using the method reac_current"

        current_dens = np.zeros((self._M, self._S))

        for m in range(self._M):
            for i in self._ind_C:
                current_dens[m, i] = np.sum(self._eff_current[m, i, :])
            self._current_dens = current_dens

        return self._current_dens