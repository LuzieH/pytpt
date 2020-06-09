import numpy as np

def is_stochastic_matrix(A):
    '''
    '''
    assert (type(A) == np.ndarray and A.shape[0] == A.shape[1]), \
      'A must be a ndarray with  shape (n, n)'

    # check that it is row-stochastic
    if not np.isclose(np.sum(A,1), np.ones(A.shape[0])).all():
        return False

    return True

def is_irreducible_matrix(A):
    '''
    '''
    assert (type(A) == np.ndarray and A.shape[0] == A.shape[1]), \
      'A must be a ndarray with  shape (n, n)'

    # check that (Id + |A|)^(n-1) > 0
    n = A.shape[0]
    B = np.eye(n) + np.abs(A)
    C = B
    for i in range(n - 1):
         C = np.matmul(C, B)

    if C.all() > 0:
        return True
    else:
        return False
