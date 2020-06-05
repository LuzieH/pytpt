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
                n=0,...,N-1
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

        assert (isinstance(P(0),np.ndarray) and not isinstance(P(0),np.matrix)), \
            "The inputted transition matrix function should map time to\
                an np.ndarray and not an np.matrix"

        assert (isinstance(ind_A, np.ndarray) and isinstance(ind_B, np.ndarray)\
                and isinstance(ind_C, np.ndarray)),\
            "The index sets have to be given as np.ndarrays."

        A = set(ind_A)
        B = set(ind_B)
        C = set(ind_C)
        intersection_AB = A.intersection(B)
        complement_AB = (C.difference(A)).difference(B)
        
        assert  (len(A)>0 and len(B)>0 and len(C)>0 and \
                len(intersection_AB) == 0 and complement_AB==C),\
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
        self.q_b = None  # backward committor
        self.q_f = None  # forward committor
        self.reac_dens = None  # reactive density
        self.reac_norm_factor = None  # normalization factor 
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
        dens_n = np.zeros((self.N, self.S))

        # initial density
        dens_n[0, :] = self.init_dens

        for n in np.arange(self.N - 1):
            # compute density at next time n+1 by applying the 
            # transition matrix
            dens_n[n + 1, :] = dens_n[n, :].dot(self.P(n))

        self.dens = dens_n
        return dens_n


    def committor(self):
        '''Function that computes the forward committor q_f (probability
        that the particle at time n will next go to B rather than A) and
        backward commitor q_b (probability that the system at time n
        last came from A rather than B) for all times n in {0,...,N-1}
        '''

        q_f = np.zeros((self.N, self.S))
        q_b = np.zeros((self.N, self.S))

        # forward committor is 1 on B, 0 on A, at time N, q_f is 
        # additionally 0 on C
        q_f[self.N - 1, self.ind_B] = 1
        # backward committor is 1 on A, 0 on B, at time 0, q_b is 
        # additionally 0 on C
        q_b[0, self.ind_A] = 1

        # density at time n-1
        dens_nmin1 = self.init_dens

        # iterate through all times n, backward in time for q_f, forward 
        # in time for q_b
        for n in range(1, self.N):

            # define the restricted transition matrices at time N-n-1
            # entries from C to C
            P_CC = self.P(self.N - n - 1)[np.ix_(self.ind_C, self.ind_C)]
            # entries from C to B
            P_CB = self.P(self.N - n - 1)[np.ix_(self.ind_C, self.ind_B)]

            # compute forward committor backwards in time
            q_f[self.N -n - 1, self.ind_C] = P_CC.dot(
                q_f[self.N-n, self.ind_C]
            ) + P_CB.dot(np.ones(np.size(self.ind_B)))

            # forward committor is 1 on B, 0 on A
            q_f[self.N - n - 1, self.ind_B] = 1

            # density at time n
            dens_n = dens_nmin1.dot(self.P(n - 1))

            # ensure that when dividing by the distribution and it's 0,
            # we don't divide by zero, there is no contribution, thus we 
            # can replace the inverse by zero
            d_n_inv = dens_n[self.ind_C]
            for i in range(np.size(self.ind_C)):
                if d_n_inv[i] > 0:
                    d_n_inv[i] = 1 / d_n_inv[i]
                # else: its just zero

            # define restricted transition matrices
            P_CC = self.P(n - 1)[np.ix_(self.ind_C, self.ind_C)]
            P_AC = self.P(n - 1)[np.ix_(self.ind_A, self.ind_C)]

            # compute backward committor forward in time
            q_b[n, self.ind_C] = d_n_inv*(
                dens_nmin1[self.ind_C]*q_b[n-1, self.ind_C]
            ).dot(P_CC) + d_n_inv*dens_nmin1[self.ind_A].dot(P_AC)

            # backward committor is 1 on A, 0 on B
            q_b[n, self.ind_A] = 1

            dens_nmin1 = dens_n

        self.q_b = q_b
        self.q_f = q_f

        return self.q_f, self.q_b


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

        # density at time n
        dens_n = self.init_dens.dot(self.P(0))

        for n in range(0, self.N):
            reac_dens[n, :] = np.multiply(
                self.q_b[n, :],
                np.multiply(dens_n, self.q_f[n, :]),
            )
            # update density for next time point
            dens_n = dens_n.dot(self.P(n-1))

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

        reac_norm_factor = np.zeros(self.N)
        for n in range(0, self.N):
            reac_norm_factor[n] = np.sum(self.reac_dens[n, :])

        self.reac_norm_factor = reac_norm_factor
        return self.reac_norm_factor


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

        if self.reac_norm_factor is None:
            self.reac_norm_factor = self.reac_norm_factor()

        norm_reac_dens = np.zeros((self.N, self.S))
        for n in range(0, self.N):
            if self.reac_norm_factor[n] != 0:
                norm_reac_dens[n, :] = self.reac_dens[n, :] /\
                                       self.reac_norm_factor[n] 
            else:
                norm_reac_dens[n] = np.nan 

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

        dens_n = self.init_dens

        for n in range(self.N-1):
            # compute reactive current
            current[n,:,:] = dens_n.reshape(S, 1) * \
                            self.q_b[n, :].reshape(S, 1) * \
                                self.P(n) * self.q_f[n+1, :]

            # compute effective current
            eff_current[n,:,:] = np.maximum(np.zeros((S, S)),\
                                            current[n,:,:] - current[n,:,:].T)

            dens_n = dens_n.dot(self.P(n))

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
        the reac_norm_factor and the transition rate.
        '''

        assert self.reac_norm_factor is not None, "The normalization \
        factor first needs to be computed by using the method \
        reac_norm_factor"
        
        assert self.rate is not None, "The transition rate first needs \
        to be computed by using the method transition_rate"

        self.av_length = np.nansum(self.reac_norm_factor) \
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
        self.committor()
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
            reac_norm_factor=self.reac_norm_factor,
            norm_reac_dens=self.norm_reac_dens,
            eff_current=self.eff_current,
            rate=self.rate,
            time_av_rate=self.time_av_rate,
            av_length=self.av_length,
        )
