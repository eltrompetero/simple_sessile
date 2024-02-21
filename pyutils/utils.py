# ====================================================================================== #
# Useful functions for sessile package.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import numpy as np
import pandas as pd
import os
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

def select_points_in_box(xy, dim):
    """Only return the points within the limits of the box.

    Parameters
    ----------
    xy : ndarray
        Locations of each point.
    dim : tuple or float
        Dimension of the box (x0, y0, dx, dy) or just a float indicating a box jutting out
        from the origin.

    Returns
    -------
    ndarray
        xy of selected points
    """

    if type(dim) is float or type(dim) is int:
        dim = (0, 0, dim, dim)

    x0 = dim[0]
    x1 = dim[0] + dim[2]
    y0 = dim[1]
    y1 = dim[1] + dim[3]

    selectix = (xy[:,0]>=x0) & (xy[:,0]<=x1) & (xy[:,1]>=y0) & (xy[:,1]<=y1)

    return xy[selectix]

def log_hist(Y, bins=20, normalize=True):
    """
    Parameters
    ----------
    Y : ndarray
    bins : int or ndarray
    normalize : bool or str, True
        If True, normalize each bin by the bin width. If 'int', then normalize by the number
        of integers within the bins.

    Returns
    -------
    ndarray
    ndarray
    """
    if not hasattr(bins, '__len__'):
        bins = np.logspace(np.log10(Y.min()), np.log10(Y.max()), bins)
        # account for numerical precision errors that put values outside the defined range
        bins[-1] += 1e-10
        bins[0] -= 1e-10
    
    n = np.bincount(np.digitize(Y, bins)-1).astype(np.float64)
    if normalize==True:
        n /= np.diff(bins)*n.sum()
    elif normalize=='int':
        n /= np.floor(np.diff(bins))*n.sum()
    else:
        raise NotImplementedError

    xmid = np.exp((np.log(bins[1:])+np.log(bins[:-1]))/2)

    return n, bins, xmid

