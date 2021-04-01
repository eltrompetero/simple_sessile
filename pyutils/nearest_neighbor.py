# ====================================================================================== #
# Module for analyzing properties of nearest neighbors in forest plots.
# Author : Eddie Lee, edlee@santafe.edu
# 
#
# MIT License
# 
# Copyright (c) 2021 Edward D. Lee
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE.
# ====================================================================================== #
from warnings import warn

from .utils import *



def dist(xy):
    """Find distances to nearest neighbors amongst a set of 2d points.

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

def survival(N, boundaries, no_boundary_correction=False):
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

def pdf(N, boundaries, no_boundary_correction=False):
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

def kl(r_sample, N, boundaries, bin_width):
    """Kullback-Leibler divergence estimated between estimate from the given sample and
    the analytic approximation for random points placed in a box (Poisson process).

    Parameters
    ----------
    r_sample : ndarray
    N : int
    boundaries : float or tuple
    bin_width : float
        Spacing used to generate histogram from given sample. This must be small enough to
        allow for linear approximation of analytic form while being large enough for the
        given sample set.

    Returns
    -------
    float
        Divergence in bits.
    """
    
    ix = np.around(r_sample/bin_width).astype(int)
    binCenters, epdf = np.unique(ix, return_counts=True)
    binCenters = binCenters * bin_width + bin_width / 2
    epdf = epdf / epdf.sum()
    
    # analytic approx to box
    p = pdf(N, boundaries)
    
    return np.nansum(epdf * (np.log2(epdf) - np.log2(p(binCenters) * bin_width)))

def interp_dkl(bindx, dkl, tol=1e-2,
               first_order=False,
               return_all=False,
               poly_fit_order=2,
               **kwargs):
    """Interpolate DKL using expansion in terms of bin widths.
    
    Parameters
    ----------
    bindx : ndarray
    dkl : ndarray
        KL-estimated at each of these bin widths.
    tol : float, 1e-3
        This is the square of the sums so is quite generous.
    return_all : bool, False
    **kwargs
        For scipy.optimize.minimize.

    Returns
    -------
    ndarray
        Coefficients of fit. Last entry is log of the estimated DKL.
    dict (optional)
        Full solution from scipy.optimize.minimize.
    ndarray (optional)
    """

    from scipy.optimize import minimize
    sortix = np.argsort(bindx)
    bindx = bindx[sortix]
    dkl = dkl[sortix]

    def fit_log(x, y):
       def cost(args):
           a, b = args
           tofitpoly = (y - np.exp(a)) / (-np.exp(b) * np.log(x))
           pfit = np.polyfit(1/x, tofitpoly, poly_fit_order)
           return np.abs(y - np.polyval(pfit, 1/x)).sum()

       soln = minimize(cost, (np.log(y.min()+1), -2), **kwargs)
       a, b = soln['x']
       tofitpoly = (y - np.exp(a)) / (-np.exp(b) * np.log(x))
       soln['x'] = np.append(soln['x'], np.polyfit(1/x, tofitpoly, poly_fit_order)[::-1])
       return soln

    soln = fit_log(bindx, dkl)

    if return_all:
        return soln['x'], soln
    return soln['x']

def _first_order_dkl(x, args):
    a, b, c = args
    return -np.exp(a) * np.log(x) * (1 + b/x) + np.exp(c)

def _second_order_dkl(x, args):
    a, b, c, d = args
    return -np.exp(a) * np.log(x) * (1 + b/x + c/x**2) + np.exp(d)

def pair_correlation(xy, bins=None, bounding_box=None):
    """Neighbor density as a function of distance (r'-r) calculated within bounding box to
    all neighbors.  <n(dr)>/N, normalized no. of neighbors at distance dr. 

    Parameters
    ----------
    xy : ndarray
    bins : ndarray
    bounding_box : tuple, None
        Specify bottom left hand corner and then width and height.
    
    Returns
    -------
    ndarray
        Correlation function.
    ndarray
        Distance.
    """
    
    # construct pairwise distance matrix
    dr = pdist(xy)
    dr = squareform(dr)
    
    if not bounding_box is None:
        assert len(bounding_box)==4
        x0, x1 = bounding_box[0], bounding_box[0] + bounding_box[2]
        y0, y1 = bounding_box[1], bounding_box[1] + bounding_box[3]
        
        # only keep elements from center of box
        selectix = (xy[:,0]>x0) & (xy[:,0]<x1) & (xy[:,1]>y0) & (xy[:,1]<y1)
        dr = dr[selectix]

    density = len(dr) / (bounding_box[2] * bounding_box[3])  # points per bounding area
    avgdist = np.sqrt( 1 / (np.pi * density) )  # typical distance between two nearest pts
    dr = dr.ravel()
    dr /= avgdist  # rescale by typical distance
    
    if bins is None:
        p, bins = np.histogram(dr, bins=np.linspace(0, dr.max(), int(np.sqrt(dr.size))))
    else:
        p, bins = np.histogram(dr, bins=bins)

    r = (bins[1:] + bins[:-1]) / 2
    p = p / p[0]
    
    return p, r
