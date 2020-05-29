import numpy as np
# https://docs.scipy.org/doc/scipy-0.15.1/reference/tutorial/linalg.html
from scipy.linalg import solve
from scipy.linalg import eig

class tpt:
    '''Calculates committor probabilities and A->B transition statistics of
    discrete-time Markov chain models in stationarity.
        
    based on: 
    Weinan, E., & Vanden-Eijnden, E. (2006). Towards a theory of 
    transition paths. Journal of statistical physics, 123(3), 503.
    
    Metzner, P., Schütte, C., & Vanden-Eijnden, E. (2009). 
    Transition path theory for Markov jump processes. Multiscale 
    Modeling & Simulation, 7(3), 1192-1219.
    
    Helfmann, L., Ribera Borrell, E., Schütte, C., & Koltai, P. (2020). 
    Extending Transition Path Theory: Periodically-Driven and Finite-Time 
    Dynamics. arXiv preprint arXiv:2002.07474.
    '''
    

    def __init__(self, P,  ind_A, ind_B,  ind_C, stat_dens=None):
        '''Initialize an instance by defining the transition matrix P and
        the subsets A and B (of the statespace) between which the 
        transition statistics should be computed.
        Args:
        P: array
            irreducible and row-stochastic (rows sum to 1) transition 
            matrix of size S x S, S is the size of the state space 
            St = {1,2,...,S} 
        ind_A: array
            set of indices of the state space that belong to the set A
        ind_B: array
            set of indices of the state space that belong to the set B
        ind_C: array
            set of indices of the state space that belong to the 
            transition region C, i.e. the set C  =  St\(A u B)        
        stat_dens: array
            stationary distribution of P, normalized
            or if None, the density will be computed automatically
        '''

        assert (isinstance(P, np.ndarray) and not isinstance(P, np.matrix)), \
            "The inputted transition matrix P should be of an np.ndarray and \
                not an np.matrix"
                
        assert (isinstance(ind_A, np.ndarray) and isinstance(ind_B, np.ndarray)\
                and isinstance(ind_C, np.ndarray)),"The index sets have to be \
            given as np.ndarrays."

        
        A = set(ind_A)
        B = set(ind_B)
        C = set(ind_C)
        intersection_AB = A.intersection(B)
        complement_AB = (C.difference(A)).difference(B)
        
        assert  (len(A) > 0 and len(B) > 0 and len(C) > 0 and \
                len(intersection_AB) == 0 and complement_AB==C),\
                "A and B have to be non-empty and disjoint sets \
                such that also their complement C is non-empty."
        
        
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
        self._reac_norm_factor = None  # normalization factor 
        self._norm_reac_dens = None  # normalized reactive density
        self._current = None  # reactive current
        self._eff_current = None  # effective reactive current
        self._rate = None  # rate of transitions from A to B
        self._length = None  # mean transition length from A to B
        self._current_dens = None  # density of the effective current

        # compute the stationary density if its not given
        if self._stat_dens is None:
            self._stat_dens = self.stationary_density()

    def stationary_density(self):
        '''Computes the stationary density of the transition matrix as
        the eigenvector of P with eigenvalue 1.
        '''

        # compute eigenvectors and eigenvalues of P
        eigv, eigvc = eig(np.transpose(self._P))
        # get index of eigenvector with eigenvalue 1 (up to small numerical 
        # error)
        index = np.where(np.isclose(eigv, 1))[0]
        # normalize
        stat_dens = (
            np.real(eigvc[:, index]) / np.sum(np.real(eigvc[:, index]))
        ).flatten()

        return stat_dens

    def backward_transitions(self): 
        '''Computes the transition matrix backwards in time. Returns a
        ndarray with shape (S, S). When the stationary density in j is
        zero, the corresponding transition matrix entries (row j) are
        set to 0.
        '''
        S = self._S
        P = self._P

        # get indexed where the stationary density is not null
        stat_dens = self._stat_dens
        idx = np.where(stat_dens != 0)[0]
        
        P_back = np.zeros(np.shape(P))
        P_back[idx, :] = P.T[idx, :] * stat_dens[:] / stat_dens[idx].reshape(S, 1)
        self._P_back = P_back

        return self._P_back

    def committor(self):
        '''Function that computes the forward committor q_f
        (probability that the chain will next go to B rather than A)
        and backward commitor q_b (probability that the system last came
        from A rather than B).
        '''
        # compute backward transition matrix
        self.backward_transitions()
        
        ### forward committor
        
        # forward  transition matrix from states in C to C
        P_C = self._P[np.ix_(self._ind_C, self._ind_C)]
        # amd from C to B
        P_CB = self._P[np.ix_(self._ind_C, self._ind_B)]

        q_f = np.zeros(self._S)
        # compute forward committor on C, the transition region
        b = np.sum(P_CB, axis=1)
        
        q_f[self._ind_C] = solve(np.eye(np.size(self._ind_C)) - P_C, b)
       
        # add entries to the forward committor vector on A, B
        # (i.e. q_f is 0 on A, 1 on B)
        q_f[self._ind_B] = 1
        
        self._q_f = q_f
        
        ### backward committor
        # backward transition matrix restricted to C to C and from C to A
        P_back_C = self._P_back[np.ix_(self._ind_C, self._ind_C)]
        P_back_CA = self._P_back[np.ix_(self._ind_C, self._ind_A)]
        
        q_b = np.zeros(self._S)
        # compute backward committor on C
        a = np.sum(P_back_CA, axis=1)

        q_b[self._ind_C] = solve(np.eye(np.size(self._ind_C)) - P_back_C, a)      
 
        # add entries to committor vector on A, B
        # (i.e. q_b is 1 on A, 0 on B)
        q_b[self._ind_A] = 1

        self._q_b = q_b

        return self._q_f, self._q_b

    def reac_density(self):
        '''
        Given the forward and backward committor and the stationary
        density, we can compute the density of reactive trajectories,
        i.e. the probability to be at x in St while being reactive.
        '''
        assert self._q_f is not None, "The committor functions need \
        first to be computed by using the method committor"

        self._reac_dens = np.multiply(
            self._q_b, np.multiply(self._stat_dens, self._q_f)
        )
        return self._reac_dens

    def reac_norm_factor(self):
        '''
        This function returns the normalization factor of the reactive 
        density, i.e. the sum over S of the reactive density. 
        This is nothing but the probability to be reactive/on a 
        transition at a certain time. 
        '''
        if self._reac_dens is None:                                                          
            self._reac_dens = self.reac_density()                                                        
        
        self._reac_norm_factor = np.sum(self._reac_dens)
        return self._reac_norm_factor

    def norm_reac_density(self):
        '''Given the reactive density and its normalization factor, 
        this function returns the normalized reactive density, i.e. 
        the probability to be at x in S, given the chain
        is reactive. 
        '''
        if self._reac_dens is None:
            self._reac_dens = self.reac_density()
        if self._reac_norm_factor is None:
            self._reac_norm_factor = self.reac_norm_factor()

        self._norm_reac_dens = self._reac_dens / self._reac_norm_factor
        return self._norm_reac_dens

    def reac_current(self):
        '''Computes the reactive current current[i,j] between nodes i
        and j, as the flow of reactive trajectories from i to j during
        one time step. 
        '''
        assert self._q_f is not None, "The committor functions need \
        first to be computed by using the method committor"

        S = self._S

        # compute current (see numpy broadcasting rules)
        q_b = self._q_b.reshape(S, 1)
        stat_dens = self._stat_dens.reshape(S, 1)
        P = self._P
        q_f = self._q_f
        current = q_b * stat_dens * P * q_f
        
        # compute effective current
        eff_current = np.maximum(np.zeros((S, S)), current - current.T)
        
        self._current = current
        self._eff_current = eff_current
                        
        return self._current, self._eff_current

    def transition_rate(self):
        '''The transition rate is the average flow of reactive
        trajectories out of A, which is the same as the average rate
        into B
        '''

        assert self._current is not None, "The reactive current first \
        needs to be computed by using the method reac_current"

        self._rate = np.sum(self._current[self._ind_A, :])
        return self._rate

    def mean_transition_length(self):
        '''The mean transition length can be computed as the ration of
        the reac_norm_factor and the transition rate.
        '''

        assert self._reac_norm_factor is not None, "The normalization \
        factor first needs to be computed by using the method \
        reac_norm_factor"
        
        assert self._rate is not None, "The transition rate first needs \
        to be computed by using the method transition_rate"

        self._length = self._reac_norm_factor / self._rate
        return self._length

    def current_density(self):
        '''
        The current density in a node is the sum of effective currents 
        over all neighbours of the node.
        '''

        assert self._current is not None, "The reactive current first \
        needs to be computed by using the method reac_current"

        self._current_dens = np.sum(self._eff_current, axis=1)
        return self._current_dens

    def compute_statistics(self):
        '''
        Function that runs all methods to compute transition statistics.
        '''
        self.stationary_density()
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
            stat_dens=self._stat_dens,
            q_f=self._q_f,
            q_b=self._q_b,
            reac_norm_factor=self._reac_norm_factor,
            norm_reac_dens=self._norm_reac_dens,
            eff_current=self._eff_current,
            rate=self._rate,
            length=self._length,
        )
