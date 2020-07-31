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
        self.kmax = r_range.size - 1
        
        self.trees = []  # list of all trees in system
        self.deadTrees = []  # list of all dead trees

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

    def grow(self, dt, **kwargs):
        """Grow trees across all size classes for one time step.
        
        Parameters
        ----------
        dt : float, 1.
            Time step.
        """
        
        # all trees grow in size
        r = self.rng.rand(len(self.trees))
        # except for the largest ones that leave the system
        removeix = []

        for i, tree in enumerate(self.trees):
            # probability that tree of given size class should grow
            if r[i] <= (self.growRate[tree.size_ix] * dt):
                if tree.size_ix == self.kmax:
                    removeix.append(i)
                else:
                    tree.grow()

        counter = 0
        for ix in removeix:
            self.deadTrees.append(self.trees.pop(ix-counter))
            counter += 1

        # introduce saplings
        for i in range(self.rng.poisson(self.g0 * dt)):
            self.trees.append( Tree(self.rng.uniform(0, self.L, size=2), self.t) )

        self.t += dt

    def kill(self, dt=1, **kwargs):
        """Kill trees across all size classes for one time step.
        
        Parameters
        ----------
        dt : float, 1.
            Time step.
        **kwargs
        """
        
        r = self.rng.rand(len(self.trees))
        killedCounter = 0

        for i, tree in enumerate(self.trees):
            if r[i] < (self.deathRate[tree.size_ix] * dt):
                self.deadTrees.append( self.trees.pop(i-killedCounter).kill(self.t) )
                killedCounter += 1

    def compete_area(self, dt=1, run_checks=False):
        """Play out root area competition between trees to kill trees.

        Parameters
        ----------
        dt : float, 1.
            Time step.
        run_checks : bool, False
        """
        
        # assemble arrays of all tree coordinates and radii
        xy = np.vstack([t.xy for t in self.trees])
        r = np.array([self.rootR[t.size_ix] for t in self.trees])
        
        # must ensure that there are at least two trees to compare
        if xy.ndim==2:
            # calculate area overlap for each pair of trees
            overlapArea = jit_overlap_area(xy, r)

            if run_checks:
                if overlapArea.shape[0] > 1000:
                    warn("Many trees in sim. Area competition calculation will be slow.")
            
            # randomly kill trees depending on whether or not below total basal met rate
            killedCounter = 0
            xi = self.env_rng.rvs()  # current env status
            deathRate = self.coeffs['dep death rate'] * self.coeffs['area competition'] * dt
            area = np.pi * r**2
            for i, tree in enumerate(self.trees):
                # as an indpt pair approx just sum over all overlapping areas
                # to be precise, one should consider areas where multiple trees overlap as different, but
                # these correspond to high order interactions
                dresource = (area[i] - overlapArea[row_ix_from_utri(i, r.size)].sum() *
                             self.coeffs['sharing fraction']) * self.coeffs['resource efficiency']
                if ((self.basalMetRate[tree.size_ix] > (dresource / xi)) and (self.rng.rand() < deathRate)):
                        # remove identified tree from the ith tree size class
                        self.deadTrees.append( self.trees.pop(i-killedCounter).kill(self.t) )
                        killedCounter += 1

    def compete_light(self, dt=1, run_checks=False, **kwargs):
        """Play out light area competition between trees to kill trees.

        Parameters
        ----------
        dt : float, 1.
            Time step.
        run_checks : bool, False
        """
        
        # needs updating for tree object-based implementation
        raise NotImplementedError

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
                counter += 1
 
    def nk(self):
        """Population count per size class.
        
        Returns
        -------
        ndarray
        """
        
        nk = np.zeros(self.kmax+1, dtype=int)
        for tree in self.trees:
            nk[tree.size_ix] += 1
        return nk
    
    def sample(self, n_sample,
               dt=1,
               sample_dt=1,
               n_forests=1,
               return_trees=False,
               n_cpus=None,
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
        n_cpus : int, None
        **kwargs
        
        Returns
        -------
        ndarray
            Sample of timepoints (n_sample, n_compartments)
        ndarray
            Time.
        ndarray
            Compartments r_k.
        """
        
        if n_forests==1:
            t = np.zeros(n_sample)
            nk = np.zeros((n_sample, self.kmax+1))
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
                if self.coeffs['area competition'] and len(self.trees):
                    self.compete_area(dt)
                if self.coeffs['light competition'] and len(self.trees):
                    self.compete_light(dt)

                i += 1
            
            if return_trees:
                return nk, t, self.rRange, self.trees
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

        return [tree.copy() for tree in self.trees]

    def plot(self,
             all_trees=None,
             fig=None,
             fig_kw={'figsize':(6,6)},
             ax=None,
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
        ax: mpl.Axes, None
        plot_kw : dict, {}
        class_ix : list, None
            Tree compartment indices to show.
        show_canopy : bool, True
        show_root : bool, True
        show_center : bool, False

        Returns
        -------
        matplotlib.Figure (optional)
            Only returned if ax was not given.
        """
        
        if all_trees is None:
            all_trees = self.trees
        if ax is None:
            if fig is None:
                fig = plt.figure(**fig_kw)
            ax = fig.add_subplot(1,1,1)
            ax_given = False
        else:
            ax_given = True
        
        # canopy area
        if show_canopy:
            patches = []
            for tree in all_trees:
                xy = tree.xy
                ix = tree.size_ix
                patches.append(Circle(xy, self.rRange[ix] * np.sqrt(self.coeffs['canopy']), ec='k'))
            pcollection = PatchCollection(patches, facecolors='green', alpha=.2)
            ax.add_collection(pcollection)

        # root area
        if show_root:
            patches = []
            for tree in all_trees:
                xy = tree.xy
                ix = tree.size_ix
                patches.append(Circle(xy, np.sqrt(self.coeffs['root'] * np.pi) * self.rootR[ix]))
            pcollection = PatchCollection(patches, facecolors='brown', alpha=.15)
            ax.add_collection(pcollection)

        # centers
        if show_center:
            xy = np.vstack([t.xy for t in self.trees])
            ax.plot(xy[:,0], xy[:,1], 'k.', ms=2)
        
        # plot settings
        ax.set(xlim=(0, self.L), ylim=(0, self.L), **plot_kw)
        
        if not ax_given:
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



class Tree():
    """Tree object for keeping track of tree properties.
    """
    def __init__(self, xy, t0=0):
        """
        Parameters
        ----------
        xy : ndarray or twople
            Position of tree.
        t0 : float, 0
            Birth time.
        """

        self.xy = xy
        self.t0 = t0
        self.t = None
        self.size_ix = 0  # size class to which tree belongs

    def grow(self):
        self.size_ix += 1
    
    def kill(self, t):
        self.t = t
        return self

    def copy(self):
        tree = Tree(self.xy, self.t0)
        tree.size_ix = self.size_ix
        tree.t = self.t
        return tree
#end Tree



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
def overlap_area(d, r1, r2):
    """Given the locations and radii of two circles, calculate the amount of area overlap.
    
    Parameters
    ----------
    d : float
        Distance between centers of two circles.
    r1 : float
    r2 : float
    
    Returns
    -------
    float
    """
    
    assert r1>0 and r2>0
    
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
            d = np.sqrt((xy[i,0]-xy[j,0])**2 + (xy[i,1]-xy[j,1])**2)
            overlapArea[counter] = overlap_area(d, r[i], r[j])
            counter += 1
   
    return overlapArea

@njit
def jit_overlap_area_avoid_repeat(xy, r, overlapArea, maxd):
    """Calculate area overlap for each pair of trees.

    Parameters
    ----------
    xy : list of ndarray or tuples
        Centers of circles.
    r : ndarray
        Radii of circles.
    area : ndarray
        Entries that 0 should be calculated. Entries that are either nonzero or np.inf
        should be ignored.
    maxd : float
        Max distance permissible between two circles before we ignore future calculations.

    Returns
    -------
    ndarray
    """

    counter = 0
    for i in range(r.size-1):
        for j in range(i+1, r.size):
            d = np.sqrt((xy[i,0]-xy[j,0])**2 + (xy[i,1]-xy[j,1])**2)

            # if far apart, avoid calculation
            if d>=maxd:
                overlapArea[counter] = 0
            else:
                overlapArea[counter] = overlap_area(d, r[i], r[j])
            counter += 1
   
    return overlapArea

@njit
def delete_flat_dist_rowcol(dist, remove_ix, n):
    """Remove elements from flattened square distance matrix corresponding to both col and
    row of specified element.

    Parameters
    ----------
    dist : ndarray
    remove_ix : int
    n : int
        Dimension of square matrix corresponding to dist.

    Returns
    -------
    ndarray
    """
    
    newDist = np.zeros((n-1) * (n-2) // 2)

    counter = 0
    inCounter = 0
    for i in range(n-1):
        for j in range(i+1, n):
            if i!=remove_ix and j!=remove_ix:
                newDist[inCounter] = dist[counter]
                inCounter += 1
            counter += 1

    return newDist

@njit
def append_flat_dist_rowcol(dist, fillval, n):
    """Append to flattened square distance matrix an additional element col and
    row of specified element.

    Parameters
    ----------
    dist : ndarray
    fillval : float
    n : int
        Dimension of square matrix corresponding to dist.

    Returns
    -------
    ndarray
    """
    
    newDist = np.zeros((n+1) * n // 2)

    counter = 0
    inCounter = 0
    for i in range(n):
        for j in range(i+1, (n+1)):
            if j==n:
                newDist[inCounter] = fillval
                inCounter += 1
            else:
                newDist[inCounter] = dist[counter]
                inCounter += 1
                counter += 1

    return newDist

