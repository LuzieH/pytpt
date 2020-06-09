import numpy as np
from inspect import isfunction
#from scipy.linalg import solve


class tpt:
    '''Calculates committor probabilities and A->B transition statistics of
    time-homogeneous or time-inhomogeneous Markov chain models over a
    finite time interval {0,...,N-1} of size N.

    based on:
    Helfmann, L., Ribera Borrell, E., Schütte, C., & Koltai, P. (2020).
    Extending Transition Path Theory: Periodically-Driven and Finite-Time
    Dynamics. arXiv preprint arXiv:2002.07474.
    '''

    def __init__(self, P, N, ind_A, ind_B,  ind_C, init_dens):
        '''Initialize an instance by defining the transition matrix and
        the sets A and B between which the transition statistics should be
        computed.

        Args:
        P: array
            - if the dynamics are time-independent:
                irreducible and row-stochastic (rows sum to 1)
                transition matrix of size S x S, S is the size
                of the state space St={1,2,...,S}
            - if the dynamics are time-dependent:
                function P(n) is a transition matrix defined for
                n=0,...,N-2
        N: int
            size of the time interval {0,1,...,N-1}
        ind_A: array
            set of indices of the state space that belong to the set A
        ind_B: array
            set of indices of the state space that belong to the set B
        ind_C: array
            set of indices of the state space that belong to the
            transition region C, i.e. the set C = St-(A u B)
        init_dens: array
            initial density at time 0
        '''
        assert (isfunction(P) or isfunction(P.func)), "The transition \
            matrices need to be inputted as a function mapping time to \
            the corresponding transition matrix."

        assert (isinstance(P(0), np.ndarray) and not isinstance(P(0),np.matrix)), \
            "The inputted transition matrix function should map time to \
             an np.ndarray and not an np.matrix"

        assert (isinstance(ind_A, np.ndarray) and
                isinstance(ind_B, np.ndarray) and
                isinstance(ind_C, np.ndarray)), \
            "The index sets have to be given as np.ndarrays."

        A = set(ind_A)
        B = set(ind_B)
        C = set(ind_C)
        intersection_AB = A.intersection(B)
        complement_AB = (C.difference(A)).difference(B)

        assert  (len(A) > 0 and
                 len(B) > 0 and
                 len(C) > 0 and
                 len(intersection_AB) == 0 and
                 complement_AB==C), \
            "A and B have to be non-empty and disjoint sets \
             such that also their complement C is non-empty."

        self.init_dens = init_dens
        self.ind_A = ind_A
        self.ind_B = ind_B
        self.ind_C = ind_C
        self.N = N
        self.P = P
        self.S = np.shape(P(0))[0]  # size of the state space

        self.dens = None # density
        self.P_back = None # backward transition matrix
        self.q_b = None  # backward committor
        self.q_f = None  # forward committor
        self.reac_dens = None  # reactive density
        self.reac_norm_fact = None  # normalization factor
        self.norm_reac_dens = None  # normalized reactive density
        self.current = None  # reactive current
        self.eff_current = None  # effective reactive current
        self.rate = None  # rate of transitions from A to B
        self.av_length = None  # mean transition length from A to B
        # time-averaged rate of transitions from A to B
        self.time_av_rate = None
        self.current_dens = None  # density of the effective current


    def density(self):
        '''Function that computes and returns an array containing the
        probability to be at time n in node i, the first index of the
        returned array is time n, the second is space/the node i.
        '''
        dens = np.zeros((self.N, self.S))

        # initial density
        dens[0, :] = self.init_dens

        # compute density at time n+1 by applying the transition matrix
        # to the density at time n
        for n in np.arange(self.N - 1):
            dens[n + 1, :] = dens[n, :].dot(self.P(n))

        self.dens = dens
        return dens


    def backward_transitions(self):
        '''Computes the transition matrix backwards in time. Returns a
        function that for each time n=1,...,N-1 assigns the backward transition
        matrix at time n. When the stationary density in j is zero, the
        corresponding transition matrix entries (row j) are set to 0.
        '''
        P_back_n = np.zeros((self.N, self.S, self.S))

        # compute backward transition matrix at each time n
        for n in range(1, self.N):
            idx = np.where(self.dens[n, :] != 0)[0]
            P_back_n[n, idx, :] = self.P(n - 1).T[idx, :] \
                                * self.dens[n - 1, :] \
                                / self.dens[n, idx].reshape(np.size(idx), 1)

        # store backward matrix in a function that assigns each time point
        # to the corresponding transition matrix
        def P_back(n):
            return P_back_n[n, :, :]

        self.P_back = P_back

        return P_back


    def forward_committor(self):
        '''Function that computes the forward committor q_f (probability
        that the process at time n will next go to B rather than A) for
        all time n in {0,..., N-1}
        '''
        q_f = np.zeros((self.N, self.S))

        # forward committor at time n=N is 1 on B and 0 on B^c
        q_f[self.N - 1, self.ind_B] = 1

        # iterate backward in time
        for n in np.flip(np.arange(0, self.N - 1)):
            # define the restricted transition matrices at time n
            P_CC = self.P(n)[np.ix_(self.ind_C, self.ind_C)]
            P_CB = self.P(n)[np.ix_(self.ind_C, self.ind_B)]

            # compute forward committor in C
            q_f[n, self.ind_C] = P_CC.dot(q_f[n + 1, self.ind_C]) \
                               + np.sum(P_CB, axis=1)

            # forward committor is 1 on B and 0 on A
            q_f[n, self.ind_B] = 1

        self.q_f = q_f
        return self.q_f


    def backward_committor(self):
        '''Function that computes the backward committor q_b (probability
        that the process at time n last came from A rather than B) for
        all time n in {0,..., N-1}
        '''
        q_b = np.zeros((self.N, self.S))

        # backward committor at time n=0 is 1 on A and 0 on A^c
        q_b[0, self.ind_A] = 1

        # iterate forward in time
        for n in range(1, self.N):

            # define restricted backward transition matrices at time n-1
            P_back_CC = self.P_back(n)[np.ix_(self.ind_C, self.ind_C)]
            P_back_CA = self.P_back(n)[np.ix_(self.ind_C, self.ind_A)]

            # compute backward committor at C
            q_b[n, self.ind_C] = P_back_CC.dot(q_b[n - 1, self.ind_C]) \
                               + np.sum(P_back_CA, axis=1)

            # backward committor is 1 on A, 0 on B
            q_b[n, self.ind_A] = 1

        self.q_b = q_b
        return self.q_b


    def reac_density(self):
        '''Given the forward and backward committor and the density,
        we can compute the density of reactive trajectories,
        i.e. the probability to be in a state in S at time n=0,...,N-1
        while being reactive.
        The function returns an array of the reactive
        density for each time (with time as the first index of the
        array).
        '''

        assert self.q_f is not None, "The committor functions need \
        first to be computed by using the method committor"

        reac_dens = np.zeros((self.N, self.S))

        for n in range(0, self.N):
            reac_dens[n, :] = np.multiply(
                self.q_b[n, :],
                np.multiply(self.dens[n], self.q_f[n, :]),
            )

        self.reac_dens = reac_dens
        return self.reac_dens


    def reac_norm_factor(self):
        '''
        This function returns the normalization factor of the reactive
        density, i.e. for each time n it returns the sum over S of
        the reactive density at that time. This is nothing but the
        probability to be reactive/on a transition at time m.
        Note that at times n=0 and N-1, the normalization factor is 0,
        since there are no reactive trajectories yet.
        '''
        if self.reac_dens is None:
            self.reac_dens = self.reac_density()

        self.reac_norm_fact = np.sum(self.reac_dens, axis=1)
        return self.reac_norm_fact


    def norm_reac_density(self):
        '''Given the reactive density and its normalization factor,
        this function returns the normalized reactive density, i.e.
        the probability to be at x in S at time n, given the chain
        is reactive.
        The function returns an array of the reactive
        density for each time (with time as the first index of the
        array).
        At times n=0 and n=N-1 the method returns None because
        the normalized density is 0 for these times, and the
        normalized reactive density thus can't be computed.
        '''

        if self.reac_dens is None:
            self.reac_dens = self.reac_density()

        if self.reac_norm_fact is None:
            self.reac_norm_fact = self.reac_norm_factor()

        norm_reac_dens = np.zeros((self.N, self.S))

        # at the time where reac_norm_fact is not null
        idx = np.where(self.reac_norm_fact != 0)[0]
        norm_reac_dens[idx, :] = self.reac_dens[idx, :] \
                                / self.reac_norm_fact[idx].reshape(np.size(idx), 1)

        # otherwise
        idx = np.where(self.reac_norm_fact == 0)[0]
        norm_reac_dens[idx, :] = np.nan

        # obs: at time 0 and N-1, the reactive density is zero, the event "to
        # be reactive" is not possible

        self.norm_reac_dens = norm_reac_dens
        return self.norm_reac_dens


    def reac_current(self):
        '''Computes the reactive current current[i,j] between nodes i at
        time n and j at time n+1, as the flow of reactive trajectories
        from i to j during one time step. Only defined for n=0,..,N-2
        '''

        assert self.q_f is not None, "The committor functions  need \
        first to be computed by using the method committor"

        S = self.S

        current = np.zeros((self.N, S, S))
        eff_current = np.zeros((self.N, S, S))

        for n in range(self.N - 1):
            # compute reactive current
            current[n, :, :] = self.dens[n].reshape(S, 1) \
                             * self.q_b[n, :].reshape(S, 1) \
                             * self.P(n) \
                             * self.q_f[n + 1, :]

            # compute effective current
            eff_current[n, :, :] = np.maximum(
                np.zeros((S, S)),
                current[n, :, :] - current[n, :, :].T,
            )

        # reactive and effective current not defined at time N-1
        current[self.N - 1] = np.nan
        eff_current[self.N - 1] = np.nan

        self.current = current
        self.eff_current = eff_current

        return self.current, self.eff_current


    def transition_rate(self):
        '''The transition rate is the average flow of reactive
        trajectories out of A (first row), or into B (second row).
        The time-averaged transition rate is the averaged transition rate over
        {0, ..., N-1}. This method returns a tuple with the transition
        rate array and the time averaged transition rate array
        '''

        assert self.current is not None, "The reactive current first needs \
        to be computed by using the method reac_current"

        rate = np.zeros((self.N, 2))

        rate[:self.N - 1, 0] = np.sum(
            self.current[:self.N - 1, self.ind_A, :], axis=(1, 2)
        )
        rate[self.N - 1, 0] = np.nan

        rate[1:, 1] = np.sum(
            self.current[:self.N - 1, :, self.ind_B], axis=(1, 2)
        )
        rate[0, 1] = np.nan

        # averaged rate over the time interval
        time_av_rate = np.zeros(2)
        time_av_rate[0] = sum(rate[:self.N - 1, 0]) / (self.N)
        time_av_rate[1] = sum(rate[1:, 1]) / (self.N)

        self.rate = rate
        self.time_av_rate = time_av_rate
        return self.rate, self.time_av_rate

    def mean_transition_length(self):
        '''The mean transition length can be computed as the ration of
        the reac_norm_fact and the transition rate.
        '''

        assert self.reac_norm_fact is not None, "The normalization \
        factor first needs to be computed by using the method \
        reac_norm_factor"

        assert self.rate is not None, "The transition rate first needs \
        to be computed by using the method transition_rate"

        self.av_length = np.nansum(self.reac_norm_fact) \
                       / np.nansum(self.rate[:, 0])

        return self.av_length


    def current_density(self):
        '''The current density in a node is the sum of effective
        currents over all neighbours of the node.
        '''

        assert self.current is not None, "The reactive current first needs \
        to be computed by using the method reac_current"

        current_dens = np.zeros((self.N, self.S))
        for n in range(self.N):
            if np.isnan(self.eff_current[n]).any():
                current_dens[n] = np.nan
            else:
                for i in self.ind_C:
                    current_dens[n, i] = np.sum(self.eff_current[n, i, :])

        self.current_dens = current_dens
        return self.current_dens

    def compute_statistics(self):
        '''
        Function that runs all methods to compute transition statistics.
        '''
        self.density()
        self.backward_transitions()
        self.forward_committor()
        self.backward_committor()
        self.norm_reac_density()
        self.reac_current()
        self.transition_rate()
        self.mean_transition_length()

    def save_statistics(self, npz_path):
        '''
        Method that saves all the computed transition statistics,
        the not computed statistics are saved as None.

        Args:

        '''
        np.savez(
            npz_path,
            dens=self.dens,
            q_f=self.q_f,
            q_b=self.q_b,
            reac_norm_fact=self.reac_norm_fact,
            norm_reac_dens=self.norm_reac_dens,
            eff_current=self.eff_current,
            rate=self.rate,
            time_av_rate=self.time_av_rate,
            av_length=self.av_length,
        )
