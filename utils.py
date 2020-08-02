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

def nearest_neighbor_dist(xy):
    """Find distances to nearest neighbors in tree plot.

    Parameters
    ----------
    xy : ndarray
        List of xy positions by row.
    
    Returns
    -------
    ndarray 
    """
    
    assert len(xy) < 1e4, "No. of pairwise distances to compute is too large."

    # calculate distance to nearest neighbor
    dr = pdist(xy)

    dr = squareform(dr)
    dr[np.diag_indices_from(dr)] = np.inf

    return dr.min(0)

def nn_survival(N, boundaries, no_boundary_correction=False):
    """Probability that nearest neighbor distance is greater than dr, or survival
    probability distribution, for randomly distributed points.

    Parameters
    ----------
    N : int
        Number of points.
    boundaries : float or tuple
        Length of one side of square plot or length of each boundary.
    no_boundary_correction : bool, False

    Returns
    -------
    function
    """

    if hasattr(boundaries, '__len__'):
        assert len(boundaries)==2
        A = boundaries[0] * boundaries[1]
        boundaries = sum(boundaries)
    else:
        A = boundaries**2
        boundaries *= 4

    rho = N/A

    if no_boundary_correction:
        def ccdf(r, rho=rho):
            return np.exp(-np.pi * rho * r**2)
        
        return ccdf

    f = boundaries * np.sqrt(1 / rho / np.pi) / 2 / A
    def ccdf(r, rho=rho, f=f):
        return (1-f) * np.exp(-np.pi * rho * r**2) + f * np.exp(-np.pi * rho * r**2 / 2)
    
    return ccdf

def nn_p(N, boundaries, no_boundary_correction=False):
    """Return probability distribution of distance to nearest neighbor. See nn_survival().

    Parameters
    ----------
    N : int
        Number of points.
    A : int
        Area of plot.
    boundaries : float or tuple
        Length of one side of square plot or length of each boundary.

    Returns
    -------
    function
    """

    if hasattr(boundaries, '__len__'):
        assert len(boundaries)==2
        A = boundaries[0] * boundaries[1]
        boundaries = sum(boundaries)
    else:
        A = boundaries**2
        boundaries *= 4

    rho = N/A

    if no_boundary_correction:
        def p(r, rho=rho):
            return 2 * np.pi * rho * r * np.exp(-np.pi * rho * r**2)
        return p

    f = boundaries * np.sqrt(1 / rho / np.pi) / 2 / A
    def p(r, rho=rho, f=f):
        return (2 * np.pi * rho * r * (1-f) * np.exp(-np.pi * rho * r**2) +
                np.pi * rho * r * f * np.exp(-np.pi * rho * r**2 / 2))
    return p

