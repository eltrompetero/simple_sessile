# ====================================================================================== #
# Automata compartment model for forest growth.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from scipy.spatial.distance import squareform
from warnings import warn
from misc.stats import PowerLaw
from .utils import *



class Forest2D():
    def __init__(self, L, g0, r_range, coeffs, nu=2, tol=.1, rng=None):
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
        nu : float, 2
            Exponent for fluctuations in environment.
        tol : float, .1
            Max value desirable for rate to probability mapping. This should be as small
            as possible to keep Poisson assumption accurate, but will slow down
            simulation when smaller.
        rng : np.random.RandomState, None
        """
        
        assert g0>=1
        assert r_range.min()>0
        assert 0<tol<1
        
        self.L = L
        self.g0 = g0
        self.tol = tol
        self.t = 0  # time counter of total age of forest
        
        self.rRange = r_range
        self.coeffs = coeffs
        self.kmax = r_range.size
        
        # each element in trees is for all trees of the same size
        # within each size class we have a list for locations and birth times
        self.trees = [[[],[]] for k in range(self.kmax)]
        self.N = 0

        # env fluctuation
        assert nu>=2
        self.nu = nu
        self.env_rng = PowerLaw(nu)
        
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

        # canopy area
        self.canopyR = coeffs.get('canopy r', 0) * rRange**(3/4)

        # canopy height
        self.canopyH = coeffs.get('canopy h', 0) * rRange**(2/3)
        
        # growth
        self.growRate = coeffs['grow'] * rRange**(1/3) / self.dx
        
        # natural mortality
        self.deathRate = coeffs['death'] * rRange**(-2/3)

        # basal metabolic rate
        self.basalMetRate = coeffs.get('basal', 0) * rRange**1.8

        if not 'area competition' in coeffs.keys():
            coeffs['area competition'] = 0.
        if not 'light competition' in coeffs.keys():
            coeffs['light competition'] = 0.

    def check_dt(self, dt):
        """Pre-simulation check that given time step will not break assumption about rates
        as probabilities from Poisson distribution. This is insufficient for checking
        competition rates since those are determined during runtime.

        Parameters
        ----------
        dt : float

        Returns
        -------
        list
            False values indicate checks were passsed for growth rate and mortality,
            respectively.
        """
        
        checks = []
        
        # growth
        if not ((self.growRate * dt)<=self.tol).all():
            checks.append( (self.growRate*dt).max() )
        else:
            checks.append(False)
       
        # mortality
        if not ((self.deathRate * dt)<=self.tol).all():
            checks.append( (self.deathRate*dt).max() )
        else:
            checks.append(False)

        return checks

    def grow(self, dt, noisy=False, **kwargs):
        """Grow trees across all size classes for one time step.
        
        This is the wrapper for multiple methods.
        
        Parameters
        ----------
        dt : float
            Time step.
        noisy : bool, False
            If True, then use Poisson distribution to sample from rates.
        dt : float, 1.
            Time step.
        """
        
        if noisy:
            self._grow_noisy(dt, **kwargs)
        else:
            self._grow_no_noise(dt, **kwargs)
            
    def _grow_noisy(self, dt=1, **kwargs):
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
            n = min(self.rng.poisson(self.growRate[k] * dt * len(self.trees[k][0])),
                    len(self.trees[k][0]))

            # select n random trees to move up a class
            randix = self._random_trees(k, n)
            for i, ix in enumerate(randix):
                self.trees[k][0].pop(ix-i)
                self.trees[k][1].pop(ix-i)
                self.N -= 1
        
        # grow all other trees besides the largest one
        for k in range(self.kmax-2, -1, -1):
            if len(self.trees[k][0]):
                # typical number of trees from a given size class that should 
                # grow is given by Poisson distribution
                n = min(self.rng.poisson(self.growRate[k] * dt * len(self.trees[k][0])),
                        len(self.trees[k][0]))

                # select n random trees to move up a class
                randix = self._random_trees(k, n)
                for i, ix in enumerate(randix):
                    self.trees[k+1][0].append( self.trees[k][0].pop(ix-i) )
                    self.trees[k+1][1].append( self.trees[k][1].pop(ix-i) )
        
        # grow saplings
        for i in range(self.rng.poisson(self.g0 * dt)):
            self.trees[0][0].append(self.rng.uniform(0, self.L, size=2))
            self.trees[0][1].append(self.t)
            self.N += 1
            
        self.t += dt

    def _grow_no_noise(self, dt=1, **kwargs):
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
            # grow is given by Poisson distribution with this average
            # this only works well when the numbers are big for discretization not to matter
            n = int(self.growRate[k] * dt * len(self.trees[k][0]))

            # select n random trees to move up a class
            randix = self._random_trees(k, n)
            for i, ix in enumerate(randix):
                self.trees[k][0].pop(ix-i)
                self.trees[k][1].pop(ix-i)
                self.N -= 1
        
        # grow all other trees besides the largest one
        for k in range(self.kmax-2, -1, -1):
            if len(self.trees[k][0]):
                # typical number of trees from a given size class that should 
                # grow is given by Poisson distribution
                n = int(self.growRate[k] * dt * len(self.trees[k][0]))

                # select n random trees to move up a class
                randix = self._random_trees(k, n)
                for i, ix in enumerate(randix):
                    self.trees[k+1][0].append(self.trees[k][0].pop(ix-i))
                    self.trees[k+1][1].append(self.trees[k][1].pop(ix-i))
        
        # grow saplings
        for i in range(int(self.g0 * dt)):
            self.trees[0][0].append(self.rng.uniform(0, self.L, size=2))
            self.trees[0][1].append(self.t)
            self.N += 1
            
        self.t += dt

    def kill(self, dt=1, noisy=False, **kwargs):
        """Kill trees across all size classes for one time step.
        
        Parameters
        ----------
        dt : float, 1.
            Time step.
        """
        
        # apply mortality rate
        if noisy:
            for k in range(self.kmax):
                self._kill_trees_bin_k_noisy(k, dt)
        else:
            for k in range(self.kmax):
                self._kill_trees_bin_k_no_noise(k, dt)

    def _kill_trees_bin_k_noisy(self, k, dt=1):
        """Kill trees in bin k. Only to be called by self.kill()."""
        
        if len(self.trees[k][0]):
            n = min(self.rng.poisson(self.deathRate[k] * dt * len(self.trees[k][0])),
                    len(self.trees[k][0]))

            # select n random trees
            randix = self._random_trees(k, n)
            for i, ix in enumerate(randix):
                self.trees[k][0].pop(ix-i)
                self.trees[k][1].pop(ix-i)
                self.N -= 1

    def _kill_trees_bin_k_no_noise(self, k, dt=1):
        """Kill trees in bin k. Only to be called by self.kill()."""
        
        if len(self.trees[k][0]):
            n = min(int(self.deathRate[k] * dt * len(self.trees[k][0])), len(self.trees[k][0]))

            # select n random trees
            randix = self._random_trees(k, n)
            for i, ix in enumerate(randix):
                self.trees[k][0].pop(ix-i)
                self.trees[k][1].pop(ix-i)
                self.N -= 1
                
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

    def compete_area(self, dt=1, run_checks=False, **kwargs):
        """Play out root area competition between trees to kill trees.

        Parameters
        ----------
        dt : float, 1.
            Time step.
        run_checks : bool, False
        """
        
        # assemble arrays of all tree coordinates and radii
        xy = np.vstack([t[0] for t in self.trees if len(t[0])])
        r = np.concatenate([[self.rootR[i]]*len(t[0])
                             for i, t in enumerate(self.trees)])
        assert len(xy)==len(r)
        
        # calculate area overlap for each pair of trees
        overlapArea = jit_overlap_area(xy, r)

        if run_checks:
            if overlapArea.shape[0]>1000:
                warn("Many trees in sim. Area competition calculation will be slow.")
        
        # randomly kill trees depending on whether or not below total basal met rate
        counter = 0
        xi = self.env_rng.rvs()  # current env status
        for i, trees in enumerate(self.trees):  # size compartments
            killedCounter = 0
            area = np.pi * self.rootR[i]**2
            for j in range(len(trees[0])):  # trees within each compartment
                # as an indpt pair approx just sum over all overlapping areas
                # technically, one should consider areas where multiple trees overlap as different
                dresource = (area - overlapArea[row_ix_from_utri(counter, r.size)].sum() *
                             self.coeffs['sharing fraction'] ) * self.coeffs['resource efficiency']
                if (self.basalMetRate[i] > (dresource / xi) and
                    self.rng.rand() < (self.coeffs['dep death rate']*self.coeffs['area competition']*dt)):
                    # remove identified tree from the ith tree size class
                    self.trees[i][0].pop(j-killedCounter)
                    self.trees[i][1].pop(j-killedCounter)
                    killedCounter += 1

                    # keep track of tot number of trees
                    self.N -= 1
                counter += 1

        return overlapArea

    def compete_light(self, dt=1, run_checks=False, **kwargs):
        """Play out light area competition between trees to kill trees.

        Parameters
        ----------
        dt : float, 1.
            Time step.
        run_checks : bool, False
        """
        
        # assemble arrays of all tree coordinates and radii
        xy = np.vstack([t[0] for t in self.trees if len(t[0])])
        r = np.concatenate([[self.canopyR[i]]*len(t[0])
                             for i, t in enumerate(self.trees)])
        h = np.concatenate([[self.canopyH[i]]*len(t[0])
                             for i, t in enumerate(self.trees)])
        if run_checks:
            assert len(xy)==len(r)==len(h)
        
        # calculate area overlap for each pair of trees
        overlapArea = jit_overlap_area(xy, r)
        # turn this overlap area into a competition rate
        overlapArea *= self.coeffs['light competition'] * dt

        if run_checks:
            if overlapArea.shape[0]>1000:
                warn("Many trees in sim. Area competition calculation will be slow.")
            if (overlapArea > self.tol).any():
                warn("Competition rate could exceed rate tolerance limit. Recommend shrinking dt.")

        # randomly kill trees with rate proportional to overlap and height diff
        counter = 0
        for i, trees in enumerate(self.trees):  # size compartments
            killedCounter = 0
            for j in range(len(trees[0])):  # trees within each compartment
                # height difference between trees
                dh = np.delete(h - h[counter], counter)
                dh[dh<0] = 0
                competeFactor = (overlapArea[row_ix_from_utri(counter, r.size)] * 
                                 (1 - np.exp(-self.coeffs['light atten'] * dh)))

                if (self.rng.rand(r.size-1) < competeFactor).any():
                    # remove identified trees from the ith tree size class
                    self.trees[i][0].pop(j-killedCounter)
                    self.trees[i][1].pop(j-killedCounter)
                    killedCounter += 1

                    # keep track of tot number of trees
                    self.N -= 1
                counter += 1
 
    def nk(self):
        """Population count per size class.
        
        Returns
        -------
        ndarray
        """
        
        return np.array([len(i[0]) for i in self.trees])
    
    def sample(self, n_sample, dt=1, sample_dt=1, n_forests=1, return_trees=False,
               **kwargs):
        """Sample system.
        
        Parameters
        ----------
        n_sample : int  
            Total number of samples.
        dt : int, 1
            Time step for simulation.
        sample_dt : float, 1.
            Save sampled spaced out in time by this amount. This means that the total
            number of iterations is n_sample / dt * sample_dt.
        n_forests : int, 1
            If greater than 1, sample multiple random forests at once.
        return_trees : bool, False
        **kwargs
        
        Returns
        -------
        ndarray
            Sample of timepoints (n_sample, n_compartments)
        ndarray
            Time points.
        ndarray
            Compartments r_k.
        """
        
        if n_forests==1:
            t = np.zeros(n_sample)
            nk = np.zeros((n_sample, len(self.trees)))
            i = 0
            counter = 0  # for no. of samples saved
            while counter < n_sample:
                # measure every dt, but make sure to account for potential floating point
                # precision errors
                if (i - counter * sample_dt / dt + 1e-15)>=0:
                    t[counter] = dt * i
                    nk[counter] = self.nk()
                    counter += 1

                self.grow(dt, **kwargs)
                self.kill(dt, **kwargs)
                if self.coeffs['area competition']:
                    self.compete_area(dt, **kwargs)
                if self.coeffs['light competition']:
                    self.compete_light(dt, **kwargs)

                i += 1
                
            return nk, t, self.rRange

        def loop_wrapper(args):
            forest = Forest2D(self.L, self.g0, self.rRange, self.coeffs, self.nu)
            nk, t, rk = forest.sample(n_sample, dt, sample_dt, **kwargs)
            return nk, t, rk, forest.trees

        with threadpool_limits(limits=1, user_api='blas'):
            with Pool(cpu_count()-1) as pool:
                nk, t, rk, trees = list(zip(*pool.map(loop_wrapper, range(n_forests))))

        if return_trees:
            return nk, t, rk, trees
        return nk, t, rk

    def snapshot(self):
        """Return copy of self.trees.
        """

        return [[i[0][:], i[1][:]] for i in self.trees]

    def plot(self,
             all_trees=None,
             fig=None,
             fig_kw={'figsize':(6,6)},
             plot_kw={},
             class_ix=None,
             show_canopy=True,
             show_root=True,
             show_center=False):
        """
        Parameters
        ----------
        all_trees : list, None
        fig : matplotlib.Figure, None
        fig_kw : dict, {'figsize':(6,6)}
        class_ix : list, None
            Tree compartment indices to show.
        show_canopy : bool, True
        show_root : bool, True
        show_center : bool, False

        Returns
        -------
        matplotlib.Figure
        """
        
        if all_trees is None:
            all_trees = self.trees
        if fig is None:
            fig = plt.figure(**fig_kw)
        if class_ix is None:
            class_ix = list(range(len(all_trees)))
        ax = fig.add_subplot(1,1,1)
        
        # canopy area
        if show_canopy:
            patches = []
            for i, trees in enumerate(all_trees):
                if i in class_ix:
                    for xy, t in zip(*trees):
                        patches.append(Circle(xy, self.rRange[i] * self.coeffs['canopy'], ec='k'))
            pcollection = PatchCollection(patches, facecolors='green', alpha=.2)
            ax.add_collection(pcollection)

        # root area
        if show_root:
            patches = []
            for i, trees in enumerate(all_trees):
                if i in class_ix:
                    for xy, t in zip(*trees):
                        patches.append(Circle(xy, self.rootR[i]))
            pcollection = PatchCollection(patches, facecolors='brown', alpha=.15)
            ax.add_collection(pcollection)

        # centers
        if show_center:
            for i, trees in enumerate(all_trees):
                if len(trees[0]) and i in class_ix:
                    xy = np.vstack(trees[0])
                    ax.plot(xy[:,0], xy[:,1], 'k.')
        
        # plot settings
        ax.set(xlim=(0, self.L), ylim=(0, self.L), **plot_kw)

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
@njit
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

@njit
def overlap_area(xy1, r1, xy2, r2):
    """Given the locations and radii of two circles, calculate the amount of area overlap.
    
    Parameters
    ----------
    xy1 : tuple or ndarray
    r1 : float
    xy2 : tuple or ndarray
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

@njit
def jit_overlap_area(xy, r):
    """Calculate area overlap for each pair of trees.

    Parameters
    ----------
    xy : list of ndarray or tuples
        Centers of circles.
    r : ndarray
        Radii of circles.

    Returns
    -------
    ndarray
    """

    overlapArea = np.zeros(r.size*(r.size-1)//2)
    counter = 0
    for i in range(r.size-1):
        for j in range(i+1, r.size):
            overlapArea[counter] = overlap_area(xy[i], r[i], xy[j], r[j])
            counter += 1
   
    return overlapArea
