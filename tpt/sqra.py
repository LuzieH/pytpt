import numpy as np


def sqra(u, A, beta, phi):
    """ Square-root approximation of the generator
    (of the Overdamped Langevin model)
    
    u: vector of pointwise evaluation of the potential
    A: adjacency matrix of the discretization
    beta: inverse temperature
    phi: the flux constant, determined by the temperature and the discr.
    
    author: Alexander Sikorski, sikorski@zib.de
    source: https://git.zib.de/cmd/cmdtools/blob/master/src/cmdtools/estimation/sqra.py
    """
    
    pi  = np.sqrt(np.exp(- beta * u))  # Boltzmann distribution
    pi /= np.sum(pi)

    D  = np.diag(pi)
    D1 = np.diag(1 / pi)
    Q  = phi * D1.dot(A).dot(D)
    return Q
