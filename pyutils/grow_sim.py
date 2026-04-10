# ====================================================================================== #
# Automata compartment model for sessile organism growth based on forests.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from scipy.spatial.distance import squareform
from scipy.spatial import KDTree
from warnings import warn
from misc.stats import PowerLaw
from types import LambdaType

from .utils import *



class Forest2D():
    def __init__(self, L, g0, r_range, coeffs, nu=2, tol=.1, bc='free', rng=None):
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
            Coefficients parameterizing unit conversions. Note that these should be given
            in the same way as the Mathematica simulation. They are converted to the units
            used in this simulation in .setup_bin_params().
        nu : float, 2
            Exponent for fluctuations in environment.
        tol : float, .1
            Max value desirable for rate to probability mapping. This should be as small
            as possible to keep Poisson assumption accurate, but will slow down
            simulation when smaller.
        bc : str, 'free'
            Boundary conditions: 'free' for open boundaries, 'periodic' for toroidal.
        rng : np.random.RandomState, None
        """

        assert g0>=1
        assert r_range.min()>0
        assert 0<tol<1
        assert bc in ('free', 'periodic')

        self.L = L
        self.g0 = g0
        self.bc = bc
        self.tol = tol
        self.t = 0  # time counter of total age of forest
        
        self.rRange = r_range
        self.coeffs = coeffs.copy()
        self.kmax = r_range.size - 1
        
        self.trees = []  # list of all trees in system
        self.deadTrees = []  # list of all dead trees

        # env fluctuation
        assert nu>1
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
        self.rootR = np.sqrt(coeffs.get('root', 1.) / np.pi) * rRange**(2/3)

        # canopy area
        self.canopyR = np.sqrt(coeffs.get('canopy r', 1.) / np.pi) * rRange

        # canopy height
        self.canopyH = coeffs.get('canopy h', 1.) * rRange**(2/3)
        
        # growth
        self.growRate = coeffs['grow'] * rRange**(1/3) / self.dx
        
        # natural mortality
        self.deathRate = coeffs['death'] * rRange**(-2/3)

        # basal metabolic rate
        self.basalMetRate = coeffs.get('basal', 1.) * rRange**1.8

        # light attenuation function (typically exponential or Theta function)
        if coeffs.get('ldecay type','theta')=='theta':
            self.ldecay_f = lambda dh: np.heaviside(dh-coeffs['ldecay length'], 0)
        elif coeffs['ldecay type']=='exp':
            self.ldecay_f = lambda dh: 1 - np.exp(-coeffs['ldecay length'] * np.clip(dh, 0, np.inf))
        # if custom type is given, then make sure it's a function
        elif isinstance(coeffs['ldecay type'], LambdaType):  
            self.ldecay_f = coeffs['ldecay type']
        else: raise NotImplementedError("Unrecognized light attenuation function.")

        if not 'area competition' in coeffs.keys():
            coeffs['area competition'] = 0.
        else:
            assert 'dep death rate' in coeffs.keys(), "Must specify 'dep death rate' if area competition is on."
            assert 'area competition' in coeffs.keys(), "Must specify 'area competition' if area competition is on."
            assert 'sharing fraction' in coeffs.keys(), "Must specify 'sharing fraction' if area competition is on."
            assert 'resource efficiency' in coeffs.keys(), "Must specify 'resource efficiency' if area competition is on."
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
        tuple
            (Boolean to indicate check was failed for growth rate,
             value of growth rate that failed)
            i.e., false values indicate checks were passed.
        tuple 
            (Boolean to indicate check was failed for mortality rate,
             value of mortality rate that failed) 
        """
        checks = []
        
        # growth
        if not ((self.growRate * dt)<=self.tol).all():
            checks.append( (True, (self.growRate*dt).max()) )
        else:
            checks.append((False, 0.))
       
        # mortality
        if not ((self.deathRate * dt)<=self.tol).all():
            checks.append( (True, (self.deathRate*dt).max()) )
        else:
            checks.append((False, 0.))

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

        for i, tree in enumerate(self.trees):
            # probability that tree of given size class should grow
            # there is choice for the dynamics of the largest trees, i.e. they can
            # disappear once they reach max size or they could persist til metabolic or
            # competitive death
            # To be done properly, the sim boundaries should effectively extend to
            # infinity, but is hard to do in some cases.
            if r[i] <= (self.growRate[tree.size_ix] * dt):
                if tree.size_ix < self.kmax:
                    tree.grow()
                else:
                    warn("Largest tree has reached max bin. Recommend increasing size range.")

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
            # calculate overlap area
            _L = self.L if self.bc=='periodic' else 0.
            #overlapArea = jit_overlap_area(xy, r, _L)
            # calculate area overlap using spatial index
            overlap_sum, _ = sparse_overlap_area(xy, r, _L)

            #if run_checks:
            #    if overlapArea.shape[0] > 1000:
            #        warn("Many trees in sim. Area competition calculation will be slow.")

            # randomly kill trees depending on whether or not below total basal met rate
            killedTreeIx = []
            xi = self.env_rng.rvs()  # current env status
            deathRate = self.coeffs['dep death rate'] * self.coeffs['area competition'] * dt
            area = np.pi * r**2
            for i, tree in enumerate(self.trees):
                # as an indpt pair approx just sum over all overlapping areas
                # to be precise, one should consider areas where multiple trees overlap as different, but
                # these correspond to high order interactions
                dresource = (area[i] - overlap_sum[i] *
                             self.coeffs['sharing fraction']) * self.coeffs['resource efficiency']
                #dresource = (area[i] - overlapArea[row_ix_from_utri(i, r.size)].sum() *
                #             self.coeffs['sharing fraction']) * self.coeffs['resource efficiency']
                if ((self.basalMetRate[tree.size_ix] > (dresource / xi)) and (self.rng.rand() < deathRate)):
                    killedTreeIx.append(i)

            for i, ix in enumerate(killedTreeIx):
                # remove identified tree from the ith tree size class
                self.deadTrees.append( self.trees.pop(ix-i).kill(self.t) )

    def compete_light(self, dt=1, run_checks=False, **kwargs):
        """Play out light area competition between trees to kill trees.

        Parameters
        ----------
        dt : float, 1.
            Time step.
        run_checks : bool, False
        **kwargs

        Returns
        -------
        None
        """

        # assemble arrays of all tree coordinates and radii
        xy = np.vstack([t.xy for t in self.trees])
        r = np.array([self.canopyR[t.size_ix] for t in self.trees])
        h = np.array([self.canopyH[t.size_ix] for t in self.trees])

        # calculate area overlap
        _L = self.L if self.bc=='periodic' else 0.
        overlapArea = jit_overlap_area(xy, r, _L)
        # calculate area overlap using spatial index
        #_, neighbors = sparse_overlap_area(xy, r, _L)

        # turn this overlap area into a competition rate
        overlapArea *= self.coeffs['light competition'] * dt

        if run_checks:
            if overlapArea.shape[0]>1000:
                warn("Many trees in sim. Area competition calculation will be slow.")
            if (overlapArea > self.tol).any():
                warn("Competition rate could exceed rate tolerance limit. Recommend shrinking dt.")

        # randomly kill trees with rate proportional to overlap and height diff
        killedTreeIx = []
        for i in range(len(self.trees)):
            dh = np.delete(h - h[i], i)  # height difference, neighbor - self, excepting self
            competeFactor = overlapArea[row_ix_from_utri(i, r.size)] * self.ldecay_f(dh)
            if self.rng.rand() < competeFactor.sum():
                killedTreeIx.append(i)

        for i, ix in enumerate(killedTreeIx):
            # remove identified trees from the ith tree size class
            self.deadTrees.append( self.trees.pop(ix-i).kill(self.t) )
 
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
            These go into self.grow().
        
        Returns
        -------
        ndarray
            Sample of timepoints (n_sample, n_compartments)
        ndarray
            Time.
        ndarray
            Compartments r_k.
        list of list of Tree
            For each forest, i.e. outermost list length is given by n_forests.
        """
        
        if n_forests==1:
            t = np.zeros(n_sample)
            nk = np.zeros((n_sample, self.kmax+1))
            trees = []

            i = 0
            counter = 0  # for no. of samples saved
            while counter < n_sample:
                # measure every dt, but make sure to account for potential floating point
                # precision errors
                if (i - counter * sample_dt / dt + 1e-15)>=0:
                    t[counter] = dt * i
                    nk[counter] = self.nk()
                    if return_trees:
                        trees.append(self.snapshot())
                    counter += 1
                
                self.grow(dt, **kwargs)
                if self.coeffs['death']:
                    self.kill(dt, **kwargs)
                if self.coeffs['area competition'] and len(self.trees):
                    self.compete_area(dt)
                if self.coeffs['light competition'] and len(self.trees):
                    self.compete_light(dt)

                i += 1
            
            if return_trees:
                return nk, t, self.rRange, trees
            return nk, t, self.rRange

        def loop_wrapper(args):
            # create a new forest with same parameters
            forest = Forest2D(self.L, self.g0, self.rRange, self.coeffs, self.nu, bc=self.bc)
            if return_trees:
                return forest.sample(n_sample, dt, sample_dt, return_trees=True, **kwargs)
            return forest.sample(n_sample, dt, sample_dt, **kwargs)

        with threadpool_limits(limits=1, user_api='blas'):
            with Pool(cpu_count()-1) as pool:
                if return_trees:
                    nk, t, rk, trees = list(zip(*pool.map(loop_wrapper, range(n_forests))))
                else:
                    nk, t, rk = list(zip(*pool.map(loop_wrapper, range(n_forests))))

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
             show_center=False,
             center_kw={'c':'k', 'ms':2}):
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
        center_kw : str, {'c':'k', 'ms':2}

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
                if class_ix is None or ix in class_ix:
                    radius = self.canopyR[ix]
                    for gxy in _ghost_positions(xy, radius, self.L, self.bc):
                        patches.append(Circle(gxy, radius, ec='k'))
            pcollection = PatchCollection(patches, facecolors='green', alpha=.2)
            ax.add_collection(pcollection)

        # root area
        if show_root:
            patches = []
            for tree in all_trees:
                xy = tree.xy
                ix = tree.size_ix
                if class_ix is None or ix in class_ix:
                    radius = self.rootR[ix]
                    for gxy in _ghost_positions(xy, radius, self.L, self.bc):
                        patches.append(Circle(gxy, radius))
            pcollection = PatchCollection(patches, facecolors='brown', alpha=.15)
            ax.add_collection(pcollection)

        # centers
        if show_center:
            if class_ix is None:
                xy = np.vstack([t.xy for t in all_trees])
            else:
                xy = np.vstack([t.xy for t in all_trees if t.size_ix in class_ix])
            ax.plot(xy[:,0], xy[:,1], '.', **center_kw)
        
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
def _ghost_positions(xy, radius, L, bc):
    """Return list of positions at which to draw a circle for plotting.

    For free BC, returns [xy]. For periodic BC, adds ghost copies shifted by
    +/-L for circles that overlap a domain edge.

    Parameters
    ----------
    xy : ndarray, (2,)
    radius : float
    L : float
    bc : str

    Returns
    -------
    list of ndarray
    """

    positions = [xy]
    if bc != 'periodic':
        return positions

    shifts = []
    if xy[0] < radius:
        shifts.append(np.array([L, 0.]))
    elif xy[0] > L - radius:
        shifts.append(np.array([-L, 0.]))
    if xy[1] < radius:
        shifts.append(np.array([0., L]))
    elif xy[1] > L - radius:
        shifts.append(np.array([0., -L]))

    for s in list(shifts):
        positions.append(xy + s)

    # corner ghosts: if near both an x and y edge, need diagonal shift too
    if len(shifts) == 2:
        positions.append(xy + shifts[0] + shifts[1])

    return positions

@njit
def pair_dist(xy_i, xy_j, L=0.):
    """Euclidean or toroidal pairwise distance.

    Parameters
    ----------
    xy_i : ndarray
    xy_j : ndarray
    L : float, 0.
        Domain side length. If >0, use minimum-image convention (periodic).
        If 0, use standard Euclidean distance.

    Returns
    -------
    float
    """

    dx = abs(xy_i[0] - xy_j[0])
    dy = abs(xy_i[1] - xy_j[1])
    if L > 0.:
        dx = min(dx, L - dx)
        dy = min(dy, L - dy)
    return np.sqrt(dx**2 + dy**2)

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

def _sparse_overlap_area(xy, r, L=0.):
    """Calculate overlap areas using a spatial index to avoid O(n^2) pair enumeration.

    Uses KDTree to find candidate pairs within interaction range, then computes
    geometric overlap only for those pairs.

    Parameters
    ----------
    xy : ndarray, (n, 2)
        Centers of circles.
    r : ndarray, (n,)
        Radii of circles.
    L : float, 0.
        Domain side length. If >0, use toroidal (periodic) distance via
        KDTree(boxsize=L).

    Returns
    -------
    overlap_sum : ndarray, (n,)
        Total overlap area for each circle summed over all neighbors.
    neighbors : dict of {int: list of (int, float)}
        For each circle i, list of (j, overlap_area) pairs with nonzero overlap.
    """

    n = len(r)
    r_max = r.max()
    # KDTree supports periodic boundaries natively via boxsize
    kd = KDTree(xy, boxsize=L) if L > 0. else KDTree(xy)

    # conservative cutoff: two circles can only overlap if dist < r_i + r_j <= 2*r_max
    candidate_pairs = kd.query_pairs(2 * r_max)

    overlap_sum = np.zeros(n)
    neighbors = {i: [] for i in range(n)}

    for i, j in candidate_pairs:
        d = pair_dist(xy[i], xy[j], L)
        a = overlap_area(d, r[i], r[j])
        if a > 0:
            overlap_sum[i] += a
            overlap_sum[j] += a
            neighbors[i].append((j, a))
            neighbors[j].append((i, a))

    return overlap_sum, neighbors

def sparse_overlap_area(xy, r, L=0.):
    n = len(r)
    r_max = r.max()
    kd = KDTree(xy, boxsize=L) if L > 0. else KDTree(xy)

    dist_mat = kd.sparse_distance_matrix(kd, 2 * r_max, output_type='coo_matrix')

    overlap_sum = np.zeros(n)
    neighbors = {i: [] for i in range(n)}

    for i, j, d in zip(dist_mat.row, dist_mat.col, dist_mat.data):
        if j <= i:
            continue
        if d >= r[i] + r[j]:
            continue

        a = overlap_area(d, r[i], r[j])
        if a > 0:
            overlap_sum[i] += a
            overlap_sum[j] += a
            neighbors[i].append((j, a))
            neighbors[j].append((i, a))

    return overlap_sum, neighbors

@njit
def jit_overlap_area(xy, r, L=0.):
    """Calculate area overlap for each pair of trees.

    Parameters
    ----------
    xy : list of ndarray or tuples
        Centers of circles.
    r : ndarray
        Radii of circles.
    L : float, 0.
        Domain side length. If >0, use toroidal (periodic) distance.

    Returns
    -------
    ndarray
    """

    overlapArea = np.zeros(r.size*(r.size-1)//2)
    counter = 0
    for i in range(r.size-1):
        for j in range(i+1, r.size):
            d = pair_dist(xy[i], xy[j], L)
            overlapArea[counter] = overlap_area(d, r[i], r[j])
            counter += 1

    return overlapArea

# ========================= #
# Deprecated functions below
# ========================= #
@njit
def jit_overlap_area_avoid_repeat(xy, r, overlapArea, maxd, L=0.):
    """Calculate area overlap for each pair of trees. (I think this came out to be slower
    than the simple method).

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
    L : float, 0.
        Domain side length. If >0, use toroidal (periodic) distance.

    Returns
    -------
    ndarray
    """

    counter = 0
    for i in range(r.size-1):
        for j in range(i+1, r.size):
            d = pair_dist(xy[i], xy[j], L)

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

