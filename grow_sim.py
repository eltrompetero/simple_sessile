# ====================================================================================== #
# Automata compartment model for forest growth.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from scipy.spatial.distance import squareform



class Forest2D():
    def __init__(self, L, g0, r_range, coeffs, rng=None):
        """
        Parameters
        ----------
        L : float
            Forest length.
        g0 : float
            Sapling appearance rate.
        r_range : ndarray
            Bins for radius.
        coeffs : dict
            Coefficients controlling how radius gets converted into other 
            measurements via allometric scaling.
            'root' : root length
        rng : np.random.RandomState, None
        """
        
        assert g0>=1
        assert r_range.min()>0
        
        self.L = L
        self.g0 = g0
        self.t = 0  # time counter of total age of forest
        
        self.rRange = r_range
        self.coeffs = coeffs
        self.kmax = r_range.size
        
        # each element in trees is for all trees of the same size
        # within each size class we have a list for locations and birth times
        self.trees = [[[],[]] for k in range(self.kmax)]
        
        self.rng = rng or np.random.RandomState()
        
        self.setup_bin_params()
        
    def setup_bin_params(self):
        """Define parameters for each bin such as death and growth rates.
        """
        
        coeffs = self.coeffs
        rRange = self.rRange
        self.dx = rRange[1] - rRange[0]  # assuming linearly spaced bins
        
        # root areas
        self.rootR = coeffs['root'] * rRange**(2/3)
        
        # growth
        self.growRate = coeffs['grow'] * rRange**(1/3) / self.dx
        assert (self.growRate<=1).all(), (self.growRate[self.growRate>1]).max()
        
        # mortality
        self.deathRate = coeffs['death'] * rRange**(-2/3)
        assert (self.deathRate<=1).all()

    def grow(self, noisy=False, **kwargs):
        """Grow trees across all size classes for one time step.
        
        This is the wrapper for multiple methods.
        
        Parameters
        ----------
        noisy : bool, False
            If True, then use Poisson distribution to sample from rates.
        dt : float, 1.
            Time step.
        """
        
        if noisy:
            self._grow_noisy(**kwargs)
        else:
            self._grow_no_noise(**kwargs)
            
    def _grow_noisy(self, dt=1):
        """Grow trees across all size classes for one time step.
        
        Parameters
        ----------
        dt : float, 1.
            Time step.
        """
        
        # all trees grow in size
        # when the largest ones grow, they leave the system
        k = self.kmax - 1
        if len(self.trees[k][0]):
            # typical number of trees from a given size class that should 
            # grow is given by Poisson distribution
            n = self.rng.poisson(self.growRate[k] * len(self.trees[k][0]))

            # select n random trees to move up a class
            randix = self._random_trees(k, n)
            for i, ix in enumerate(randix):
                self.trees[k][0].pop(ix-i)
                self.trees[k][1].pop(ix-i)
        
        # grow all other trees besides the largest one
        for k in range(self.kmax-2, -1, -1):
            if len(self.trees[k][0]):
                # typical number of trees from a given size class that should 
                # grow is given by Poisson distribution
                n = self.rng.poisson(self.growRate[k] * len(self.trees[k][0]))

                # select n random trees to move up a class
                randix = self._random_trees(k, n)
                for i, ix in enumerate(randix):
                    self.trees[k+1][0].append( self.trees[k][0].pop(ix-i) )
                    self.trees[k+1][1].append( self.trees[k][1].pop(ix-i) )
        
        # grow saplings
        for i in range(self.rng.poisson(self.g0)):
            self.trees[0][0].append(self.rng.uniform(0, self.L, size=2))
            self.trees[0][1].append(self.t)
            
        self.t += dt

    def _grow_no_noise(self, dt=1):
        """Grow trees across all size classes for one time step.
        
        Parameters
        ----------
        dt : float, 1.
            Time step.
        """
        
        # all trees grow in size
        # when the largest ones grow, they leave the system
        k = self.kmax - 1
        if len(self.trees[k][0]):
            # typical number of trees from a given size class that should 
            # grow is given by Poisson distribution
            n = int(self.growRate[k] * len(self.trees[k][0]))

            # select n random trees to move up a class
            randix = self._random_trees(k, n)
            for i, ix in enumerate(randix):
                self.trees[k][0].pop(ix-i)
                self.trees[k][1].pop(ix-i)
        
        # grow all other trees besides the largest one
        for k in range(self.kmax-2, -1, -1):
            if len(self.trees[k][0]):
                # typical number of trees from a given size class that should 
                # grow is given by Poisson distribution
                n = int(self.growRate[k] * len(self.trees[k][0]))

                # select n random trees to move up a class
                randix = self._random_trees(k, n)
                for i, ix in enumerate(randix):
                    self.trees[k+1][0].append(self.trees[k][0].pop(ix-i))
                    self.trees[k+1][1].append(self.trees[k][1].pop(ix-i))
        
        # grow saplings
        for i in range(self.g0):
            self.trees[0][0].append(self.rng.uniform(0, self.L, size=2))
            self.trees[0][1].append(self.t)
            
        self.t += dt

    def kill(self, dt=1):
        """Kill trees across all size classes for one time step.
        
        Parameters
        ----------
        dt : float, 1.
            Time step.
        """
        
        # apply mortality rate
        for k in range(self.kmax):
            self._kill_trees(k)
                    
    def _kill_trees(self, k):
        """Kill trees in class k. Only to be called by self.kill()."""
        
        if len(self.trees[k][0]):
            n = min(self.rng.poisson(self.deathRate[k] * len(self.trees[k][0])),
                    len(self.trees[k][0]))

            # select n random trees
            randix = self._random_trees(k, n)
            for i, ix in enumerate(randix):
                self.trees[k][0].pop(ix-i)
                self.trees[k][1].pop(ix-i)
                
    def _random_trees(self, k, n, return_xy=False):
        """Select random trees from class k.
        
        Is there a faster way to execute this?
        
        Parameters
        ----------
        k : int
            Size class
        n : int
            Number of trees to select.
        return_xy : bool, False
            If True, return locations as well.
        
        Returns
        -------
        ndarray
            Sorted indices of random trees.
        list of tuple (optional)
            Coordinates of trees.
        """
        
        randix = np.sort(self.rng.choice(range(len(self.trees[k][0])), size=n, replace=False))
        if not return_xy:
            return randix

        xy = self.trees[k][0][randix]
        return randix, xy

    def compete_area(self):
        """Play out area competition between trees to kill trees.
        """
        
        # assemble arrays of all tree coordinates and radii
        xy = np.vstack([t[0] for t in self.trees if len(t[0])])
        r = np.concatenate([[self.rootR[i]]*len(t[0])
                             for i, t in enumerate(self.trees)])
        assert len(xy)==len(r)
        
        # calculate area overlap for each pair of trees
        overlapArea = np.zeros(r.size*(r.size-1)//2)
        counter = 0
        for i in range(r.size-1):
            for j in range(i+1, r.size):
                overlapArea[counter] = overlap_area(xy[i], r[i], xy[j], r[j])
                counter += 1
        overlapArea = squareform(overlapArea)

        # randomly kill trees with rate proportional to overlap with other trees
        counter = 0
        for i, trees in enumerate(self.trees):
            killix = []
            for j in range(len(trees[0])):
                if (self.rng.rand(r.size) < (overlapArea[counter] * self.coeffs['area competition'])).any():
                    killix.append(j)

                counter += 1
            
            # remove identified trees from the ith tree size class
            # looping through a second time helps keep indices in right order
            for j, ix in enumerate(killix):
                self.trees[i][0].pop(ix-j)
                self.trees[i][1].pop(ix-j)
        
        assert counter==len(overlapArea)

    def nk(self):
        """Population count per size class.
        
        Returns
        -------
        ndarray
        """
        
        return np.array([len(i[0]) for i in self.trees])
    
    def sample(self, n_sample, dt=1):
        """Sample system.
        
        Parameters
        ----------
        n_sample : int
        dt : int, 1
        
        Returns
        -------
        ndarray
            Sample of timepoints (n_sample, n_compartments)
        ndarray
            Compartments r_k.
        """
        
        nk = np.zeros((n_sample, len(self.trees)))
        counter = 0
        for i in range(n_sample*dt):
            if (i%dt)==0:
                nk[counter] = self.nk()
                counter += 1
            self.grow()
            self.kill()
            
        return nk, self.rRange

    def snapshot(self):
        """Return copy of self.trees.
        """

        return [[i[0][:], i[1][:]] for i in self.trees]

    def plot(self, all_trees=None, fig=None, fig_kw={'figsize':(6,6)}):
        """
        Parameters
        ----------
        all_trees : list, None
        fig : matplotlib.Figure, None
        fig_kw : dict, {'figsize':(6,6)}

        Returns
        -------
        matplotlib.Figure
        """
        
        if all_trees is None:
            all_trees = self.trees
        if fig is None:
            fig = plt.figure(**fig_kw)
        ax = fig.add_subplot(1,1,1)
        
        # canopy area
        patches = []
        for i, trees in enumerate(all_trees):
            for xy, t in zip(*trees):
                patches.append(Circle(xy, self.rRange[i] * self.coeffs['canopy'], ec='k'))
        pcollection = PatchCollection(patches, facecolors='green', alpha=.2)
        ax.add_collection(pcollection)

        # root area
        patches = []
        for i, trees in enumerate(all_trees):
            for xy, t in zip(*trees):
                patches.append(Circle(xy, self.rootR[i]))
        pcollection = PatchCollection(patches, facecolors='brown', alpha=.15)
        ax.add_collection(pcollection)
        
        # plot settings
        ax.set(xlim=(0, self.L), ylim=(0, self.L))

        return fig
#end Forest2D


class LogForest2D(Forest2D):
    def setup_bin_params(self):
        """Define parameters for each bin such as death and growth rates.
        """
        
        coeffs = self.coeffs
        rRange = self.rRange
        b = rRange[1] / rRange[0]  # assuming same log spacing
        self.dx = np.log(rRange[0] / np.sqrt(b)) + np.log(b) * np.arange(rRange.size+1)
        self.dx = np.exp(np.diff(self.dx))
        
        # root areas
        self.rootR = coeffs['root'] * rRange**(2/3)
        
        # growth
        self.growRate = coeffs['grow'] * rRange**(-1/3)
        assert (self.growRate<=1).all(), (self.growRate[self.growRate>1]).max()
        
        # mortality
        self.deathRate = coeffs['death'] * rRange**(-2/3)
        assert (self.deathRate<=1).all()
#end LogForest2D



# ================ #
# Useful functions 
# ================ #
def _area_integral(xbds, r):
    """Integral for area of circle centered at origin.
    
    Parameters
    ----------
    xbds : tuple
    r : float
        Radius of circle.
    """
    
    assert abs(xbds[0])<=r and abs(xbds[1])<=r
    
    def fcn(x):
        if x**2==r**2:
            return x * np.sqrt(r**2 - x**2) + r**2 * np.sign(x) * np.pi/2
        return x * np.sqrt(r**2 - x**2) + r**2 * np.arctan(x / np.sqrt(r**2 - x**2))
    
    return fcn(xbds[1]) - fcn(xbds[0])

def overlap_area(xy1, r1, xy2, r2):
    """Given the locations and radii of two circles, calculate the amount of area overlap.
    
    Parameters
    ----------
    xy1 : tuple
    r1 : float
    xy2 : tuple
    r2 : float
    
    Returns
    -------
    float
    """
    
    assert r1>0 and r2>0
    
    d = np.sqrt((xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2)
    # no overlap
    if d>=(r1+r2):
        return 0.
    # total overlap
    elif (d+min(r1,r2))<=max(r1,r2):
        return np.pi * min(r1,r2)**2
    
    # point of intersection if two circles were to share the same x-axis
    xstar = (r1**2 - r2**2 + d**2) / (2*d)
    area = _area_integral((xstar, r1), r1) + _area_integral((-r2, xstar-d), r2)
    
    return area