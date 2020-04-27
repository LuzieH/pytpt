import numpy as np

def is_stochastic_matrix(A):
    '''
    '''
    # check that it is row-stochastic
    for i in np.arange(A.shape[0]):
        if not np.isclose(np.sum(A[i, :]), 1):
            return False
    
    return True
