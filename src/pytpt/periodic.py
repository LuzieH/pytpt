import numpy as np
from inspect import isfunction
from scipy.linalg import solve
from scipy.linalg import eig


class tpt:
    '''Calculates committor probabilities and A->B transition statistics of
    Markov chain models with periodic forcing.

    based on:
    Helfmann, L., Ribera Borrell, E., Schütte, C., & Koltai, P. (2020).
    Extending Transition Path Theory: Periodically-Driven and Finite-Time
    Dynamics. arXiv preprint arXiv:2002.07474.
    '''

    def __init__(self, P, M, ind_A, ind_B,  ind_C):
        '''Initialize an instance by defining the periodically forced
        transition matrix and the sets A and B between which the transition
        statistics should be computed.

        Args:
        P:  function mapping time modulo (0,1,...,M-1) to the
            corresponding transition matrix (row-stochastic)
            of size S x S, S is the size of the state space
            St = {1,2,...,S}), moreover the product of transition
            matrices should be irreducible
        M: int
            size of the period
        ind_A: array
            set of indices of the state space that belong to the set A
        ind_B: array
            set of indices of the state space that belong to the set B
        ind_C: array
            set of indices of the state space that belong to the
            transition region C, i.e. the set C  =  St-(A u B)
        '''

        assert (isfunction(P) or isfunction(P.func)), \
            "The transition matrices need to be inputted as a function \
             mapping time to the corresponding transition matrix."

        assert np.isclose(P(0), P(M)).all(), "The transition matrix function \
            needs to the time modulo M to the corresponding transition matrix."

        assert (isinstance(P(0), np.ndarray) and not isinstance(P(0), np.matrix)), \
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

        assert (len(A) > 0 and
                len(B) > 0 and
                len(C) > 0 and
                len(intersection_AB) == 0 and
                complement_AB == C), \
            "A and B have to be non-empty and disjoint sets \
             such that also their complement C is non-empty."

        self.P = P
        self.M = M
        self.ind_A = ind_A
        self.ind_B = ind_B
        self.ind_C = ind_C
        self.S = np.shape(self.P(0))[0]  # size of the state space

        # compute the stationary density
        self.stat_dens = self.stationary_density()

        # compute the backward transitin matrix and store as function
        self.P_back = self.backward_transitions()

        self.q_b = None  # backward committor
        self.q_f = None  # forward committor
        self.reac_dens = None  # reactive density
        self.reac_norm_fact = None  # reactive normalization factor
        self.norm_reac_dens = None  # normalized reactive density
        self.current = None  # reactive current
        self.eff_current = None  # effective reactive current
        self.rate = None  # rate of transitions from A to B
        self.av_length = None  # mean transition length from A to B
        self.current_dens = None  # density of the effective current

    def stationary_density(self):
        '''Computes the periodically varying stationary densities at times
        m=0,...,M-1 of the transition matrices and returns them.
        '''

        stat_dens = np.zeros((self.M, self.S))

        # product of transition matrices over 1 period starting at 0
        P_bar = self.P(0)
        for m in np.arange(1, self.M):
            P_bar = P_bar.dot(self.P(m))

        # compute stationary density of P_bar
        eigv, eigvc = eig(np.transpose(P_bar))
        # get index of eigenvector with eigenvalue 1
        index = np.where(np.isclose(eigv, 1))[0]
        # normalize
        stat_dens[0, :] = (
            np.real(eigvc[:, index]) / np.sum(np.real(eigvc[:, index]))
        ).flatten()

        # compute remaining densities
        for m in np.arange(1, self.M):
            stat_dens[m, :] = stat_dens[m-1, :].dot(self.P(m-1))

        return stat_dens

    def backward_transitions(self):
        '''Computes the transition matrix backwards in time. Returns a
        function that for each time assigs the correct backward
        transition matrix modulo M. When the stationary density in j is
        zero, the corresponding transition matrix entries (row j) are
        set to 0.
        '''
        P_back_m = np.zeros((self.M, self.S, self.S))

        for m in range(self.M):
            # compute backward transition matrix
            idx = np.where(self.stat_dens[np.mod(m, self.M), :] != 0)[0]
            P_back_m[m, idx, :] = self.P(m-1).T[idx, :] \
                                * self.stat_dens[np.mod(m - 1, self.M), :] \
                                / self.stat_dens[np.mod(m, self.M), idx].reshape(np.size(idx), 1)

        # store backward matrix in a function that assigns each time point
        # to the corresponding transition matrix
        def P_back(k):
            return P_back_m[np.mod(k, self.M), :, :]

        return P_back

    def forward_committor(self):
        '''Function that computes the forward committor q_f (probability
        that the process will next go to B rather than A)
        of the periodic system by using the stacked
        equations.
        '''

        # dimension of sets B, C
        dim_B = np.size(self.ind_B)
        dim_C = np.size(self.ind_C)

        # first, find the forward committor q^+_0 at time 0
        # we have to solve: a q^+_0 = (I-D)q^+_0 = b (see the proof of Lemma 4.6. of our paper)
        # where D is the result of multipling the transition matrices restricted in C
        # over one period, D = P(0)|C x... x P(M-1)|C
        # and b = sum_tau^M  P(0)|C x ... x P(tau-1)|C,B x (1)

        # compute a=(I-D) and b
        D = np.diag(np.ones(dim_C))
        b = np.zeros(dim_C)
        for m in np.arange(self.M):
            P_m_CB = self.P(m)[np.ix_(self.ind_C, self.ind_B)]
            P_m_CC = self.P(m)[np.ix_(self.ind_C, self.ind_C)]
            b += D.dot(P_m_CB).dot(np.ones(dim_B))
            D = D.dot(P_m_CC)

        a = np.eye(dim_C) - D

        # initialize forward committor
        q_f = np.zeros((self.M, self.S))

        # compute q_0^+ on C
        q_f[0, self.ind_C] = solve(a, b)

        # set q_0^+ on A and B
        q_f[:, self.ind_B] = 1

        # second, compute forward committor at remaining times iteratively
        for m in np.flip(np.arange(1, self.M)):
            P_m_CS = self.P(m)[self.ind_C, :]
            q_f[m, self.ind_C] = P_m_CS.dot(q_f[np.mod(m + 1, self.M), :])

        self.q_f = q_f
        return self.q_f

    def backward_committor(self):
        '''Function that computes the backward
        committor q_b (probability that the system last came from A
        rather than B) of the periodic system by using the stacked
        equations.
        '''

        # dimension of sets A, C
        dim_A = np.size(self.ind_A)
        dim_C = np.size(self.ind_C)

        # first, find backward committor q^-_0 at time 0
        # to solve a q^-_0 = (I-D_back)q^-_0 = a
        # multiplied backward transition matrix over all times with only
        # transitions in C

        D_back = np.diag(np.ones(dim_C))
        b = np.zeros(dim_C)

        # flip loop over the period, but start at 0
        times = np.arange(1, self.M + 1)[::-1]
        times[0] = 0

        for m in times:
            P_back_m_CA = self.P_back(m)[np.ix_(self.ind_C, self.ind_A)]
            P_back_m_CC = self.P_back(m)[np.ix_(self.ind_C, self.ind_C)]

            b += D_back.dot(P_back_m_CA).dot(np.ones(dim_A))
            D_back = D_back.dot(P_back_m_CC)
        a = np.eye(dim_C) - D_back

        # initialize q^-, compute q_0^- on C and set q_0^= on A
        q_b = np.zeros((self.M, self.S))
        q_b[0, self.ind_C] = solve(a, b)
        q_b[:, self.ind_A] = 1

        # second, compute committor for remaining times iteratively
        for m in np.arange(1, self.M):
            P_back_m_CS = self.P_back(m)[self.ind_C, :]
            q_b[m, self.ind_C] = P_back_m_CS.dot(q_b[m - 1, :])

        self.q_b = q_b

        return self.q_b

    def reac_density(self):
        '''Given the forward and backward committor and the density,
        we can compute the density of reactive trajectories,
        i.e. the probability to be in a state in S at time m while
        being reactive.
        The function returns an array of the reactive
        density for each time (with time as the first index of the
        array).
        '''

        assert self.q_f is not None, "The committor functions need \
        first to be computed by using the method committor"

        reac_dens = np.zeros((self.M, self.S))
        for m in range(self.M):
            reac_dens[m, :] = np.multiply(
                self.q_b[m, :],
                np.multiply(self.stat_dens[m, :], self.q_f[m, :]),
            )
        self.reac_dens = reac_dens
        return self.reac_dens

    def reac_norm_factor(self):
        '''
        This function returns the normalization factor of the reactive
        density, i.e. for each time m it returns the sum over S of
        the reactive density at that time. This is nothing but the
        probability to be reactive/on a transition at time m.
        '''
        if self.reac_dens is None:
            self.reac_dens = self.reac_density()

        reac_norm_fact = np.zeros(self.M)
        for m in range(0, self.M):
            reac_norm_fact[m] = np.sum(self.reac_dens[m, :])

        self.reac_norm_fact = reac_norm_fact
        return self.reac_norm_fact

    def norm_reac_density(self):
        '''Given the reactive density and its normalization factor,
        this function returns the normalized reactive density, i.e.
        the probability to be at x in S at time m, given the chain
        is reactive.
        The function returns an array of the reactive
        density for each time (with time as the first index of the
        array).
        '''

        if self.reac_dens is None:
            self.reac_dens = self.reac_density()

        if self.reac_norm_fact is None:
            self.reac_norm_fact = self.reac_norm_factor()

        norm_reac_dens = np.zeros((self.M, self.S))
        for m in range(self.M):
            if self.reac_norm_fact[m] != 0:
                norm_reac_dens[m, :] = self.reac_dens[m, :] \
                                     / self.reac_norm_fact[m]
            else:
                norm_reac_dens[m, :] = np.nan

        self.norm_reac_dens = norm_reac_dens
        return self.norm_reac_dens

    def reac_current(self):
        '''Computes the reactive current current[i,j] between nodes i
        and j, as the flow of reactive trajectories from i to j during
        one time step.
        '''
        assert self.q_f is not None, "The committor functions  need \
        first to be computed by using the method committor"

        S = self.S

        current = np.zeros((self.M, S, S))
        eff_current = np.zeros((self.M, S, S))

        for m in range(self.M):
            # compute current
            current[m, :, :] = self.stat_dens[m, :].reshape(S, 1) \
                             * self.q_b[m, :].reshape(S, 1) \
                             * self.P(m) \
                             * self.q_f[np.mod(m + 1, self.M), :]

            # compute effective current
            eff_current[m, :, :] = np.maximum(
                np.zeros((S, S)),
                current[m, :, :] - current[m, :, :].T,
            )

        self.current = current
        self.eff_current = eff_current

        return self.current, self.eff_current

    def transition_rate(self):
        '''The transition rate is the average flow of reactive
        trajectories out of A at time m (first row) or into B at time m
        (second row). The time-averaged transition rate is the averaged
        transition rate (out of A and into B) over {0, ..., M-1}. This
        method returns a tuple with the transition rate array and the
        time averaged transition rate array
        '''

        assert self.current is not None, "The reactive current first \
        needs to be computed by using the method reac_current"

        rate = np.zeros((self.M, 2))

        # for each time m, sum of all currents out of A into S
        rate[:, 0] = np.sum(
            self.current[:, self.ind_A, :], axis=(1, 2)
        )

        # for each time m, sum of all currents from S into B
        rate[:, 1] = np.sum(
            self.current[:, :, self.ind_B], axis=(1, 2)
        )

        # averaged rate over the period
        time_av_rate = np.zeros(2)
        time_av_rate[0] = sum(rate[:, 0]) / (self.M) # out of A
        time_av_rate[1] = sum(rate[:, 1]) / (self.M) # into B

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

        current_dens = np.zeros((self.M, self.S))

        for m in range(self.M):
            for i in self.ind_C:
                current_dens[m, i] = np.sum(self.eff_current[m, i, :])
            self.current_dens = current_dens

        return self.current_dens

    def compute_statistics(self):
        '''
        Function that runs all methods to compute transition statistics.
        '''
        self.stationary_density()
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
            stat_dens=self.stat_dens,
            q_f=self.q_f,
            q_b=self.q_b,
            reac_norm_fact=self.reac_norm_fact,
            norm_reac_dens=self.norm_reac_dens,
            eff_current=self.eff_current,
            rate=self.rate,
            time_av_rate=self.time_av_rate,
            av_length=self.av_length,
        )
