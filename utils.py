# ====================================================================================== #
# Forest analysis.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import numpy as np
from netCDF4 import Dataset
from scipy.spatial.distance import pdist, squareform
from numpy import ma
from numba import njit
from threadpoolctl import threadpool_limits
from multiprocess import Pool, cpu_count
from statsmodels.distributions import ECDF



@njit
def row_ix_from_utri(i, n):
    """Indices that correspond to row indices of non-diagonal elements 
    of an unraveled, symmetric matrix using squareform.
    
    Parameters
    ----------
    i : int
        Row index.
    n : int
        Dimension of square matrix.
        
    Returns
    -------
    ndarray
        Indices that would give column entries of full square matrix.
    """
    
    if i==0:
        return np.array(list(range(n-1)))
    assert i<n
    
    ix = np.zeros(n-1).astype(np.int64)
    offset = 0
    for j in range(i):
        ix[j] = (i-j-1) + offset
        offset += n-j-1
    
    for j in range(i+1, n):
        ix[j-1] = offset + j - i - 1
    return ix
