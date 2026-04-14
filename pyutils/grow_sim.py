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
from numba.experimental import jitclass
from numba.typed import List
from time import perf_counter

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
        
        self.trees = List.empty_list(TREE_NB_TYPE)  # njit-compatible list of trees
        self.deadTrees = []  # list of all dead trees

        # neighbor distance cache: {tree_id: {neighbor_id: distance}}
        # only stores pairs within self.d_max
        self._nbr_dist = Dict.empty(key_type=types.int64, value_type=types.DictType(types.int64, types.float64))
        self._id_to_tree = Dict.empty(key_type=types.int64, value_type=TREE_NB_TYPE)
        self._next_id = incrementor(0)
        self._next_id_value = 0  # tracks current generator state for pickling

        # cached overlap areas: {tree_id: {neighbor_id: overlap_area}}
        # only recomputed for trees in _dirty_ids
        self._root_overlap = {}
        self._canopy_overlap = {}
        self._dirty_ids = set()

        # env fluctuation
        assert nu>1
        self.nu = nu
        self.env_rng = PowerLaw(nu)
        
        self.rng = rng or np.random.RandomState()
        
        self.setup_bin_params()
        
    def setup_bin_params(self):
        """Define parameters for each bin such as death and growth rates."""
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

        # max interaction distance: beyond this, no pair can ever overlap
        self.d_max = 2 * max(self.rootR.max(), self.canopyR.max())

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

    def __getstate__(self):
        """Convert numba-typed internals to plain Python for pickling."""

        # convert trees to plain _Tree objects
        plain_trees = [_Tree.from_tree_nb(t) for t in self.trees]
        plain_dead = []
        for t in self.deadTrees:
            if hasattr(t, 'xy'):
                plain_dead.append(_Tree(t.xy, t.t0, t.id, t.size_ix, t.t))
            else:
                plain_dead.append(t)

        # convert numba typed dicts to plain Python dicts
        plain_nbr_dist = {int(k): {int(k2): float(v2) for k2, v2 in v.items()}
                          for k, v in self._nbr_dist.items()}

        state = self.__dict__.copy()
        state['trees'] = plain_trees
        state['deadTrees'] = plain_dead
        state['_nbr_dist'] = plain_nbr_dist
        state['_id_to_tree'] = None  # rebuilt in __setstate__
        state['_next_id'] = int(self._next_id_value)
        # lambdas and unpicklable objects — rebuilt in __setstate__
        state.pop('ldecay_f', None)
        state.pop('env_rng', None)
        return state

    def __setstate__(self, state):
        """Restore from pickled state, rebuilding numba-typed internals."""

        next_id_val = state.pop('_next_id')
        plain_trees = state.pop('trees')
        plain_nbr_dist = state.pop('_nbr_dist')

        self.__dict__.update(state)

        # rebuild ldecay_f from coeffs
        coeffs = self.coeffs
        if coeffs.get('ldecay type', 'theta') == 'theta':
            self.ldecay_f = lambda dh: np.heaviside(dh - coeffs['ldecay length'], 0)
        elif coeffs['ldecay type'] == 'exp':
            self.ldecay_f = lambda dh: 1 - np.exp(-coeffs['ldecay length'] * np.clip(dh, 0, np.inf))
        elif isinstance(coeffs['ldecay type'], LambdaType):
            self.ldecay_f = coeffs['ldecay type']

        # rebuild numba typed tree list
        self.trees = List.empty_list(TREE_NB_TYPE)
        for pt in plain_trees:
            self.trees.append(pt.to_tree_nb())

        # rebuild id_to_tree
        self._id_to_tree = Dict.empty(key_type=types.int64, value_type=TREE_NB_TYPE)
        for tree in self.trees:
            self._id_to_tree[tree.id] = tree

        # rebuild numba typed nbr_dist
        self._nbr_dist = Dict.empty(key_type=types.int64,
                                     value_type=types.DictType(types.int64, types.float64))
        for tid, nbrs in plain_nbr_dist.items():
            inner = Dict.empty(key_type=types.int64, value_type=types.float64)
            for nbr_id, d in nbrs.items():
                inner[nbr_id] = d
            self._nbr_dist[tid] = inner

        # rebuild id generator
        self._next_id = incrementor(next_id_val)

        # rebuild env_rng
        self.env_rng = PowerLaw(self.nu)

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

    @staticmethod
    @njit
    def _jit_nbr_dist_add(tree, nbr_dist, L, trees, d_max):
        """Numba kernel for updating neighbor-distance cache for one new tree."""

        tid = tree.id
        nbr_dist[tid] = Dict.empty(key_type=types.int64, value_type=types.float64)
        for other in trees:
            d = pair_dist(tree.xy, other.xy, L)
            if d < d_max:
                nbr_dist[tid][other.id] = d
                nbr_dist[other.id][tid] = d

    def _nbr_dist_add(self, tree):
        """Add a tree to the neighbor distance cache.

        Must be called BEFORE appending the tree to self.trees so that
        self.trees iteration doesn't include the new tree itself.

        Parameters
        ----------
        tree : Tree
        """
        L = self.L if self.bc == 'periodic' else 0.
        self._jit_nbr_dist_add(tree, self._nbr_dist, L, self.trees, self.d_max)
        self._root_overlap[tree.id] = {}
        self._canopy_overlap[tree.id] = {}
        self._dirty_ids.add(tree.id)

    def _nbr_dist_remove(self, tree_id):
        """Remove a tree from the neighbor distance and overlap caches.

        Parameters
        ----------
        tree_id : int
        """
        # clean up overlap caches
        for nbr_id in self._root_overlap.get(tree_id, {}):
            self._root_overlap[nbr_id].pop(tree_id, None)
        self._root_overlap.pop(tree_id, None)

        for nbr_id in self._canopy_overlap.get(tree_id, {}):
            self._canopy_overlap[nbr_id].pop(tree_id, None)
        self._canopy_overlap.pop(tree_id, None)

        # clean up distance cache
        for nbr_id in self._nbr_dist[tree_id]:
            del self._nbr_dist[nbr_id][tree_id]
        del self._nbr_dist[tree_id]
        del self._id_to_tree[tree_id]

        self._dirty_ids.discard(tree_id)

    def _nbr_dist_rebuild(self):
        """Full rebuild of the neighbor distance cache from self.trees."""
        self._nbr_dist = {tree.id: Dict.empty(key_type=types.int64, value_type=types.float64) for tree in self.trees}
        self._id_to_tree = Dict.empty(key_type=types.int64, value_type=TREE_NB_TYPE)
        for tree in self.trees:
            self._id_to_tree[tree.id] = tree
        _L = self.L if self.bc == 'periodic' else 0.
        for i, t1 in enumerate(self.trees):
            for t2 in self.trees[i+1:]:
                d = pair_dist(t1.xy, t2.xy, _L)
                if d < self.d_max:
                    self._nbr_dist[t1.id][t2.id] = d
                    self._nbr_dist[t2.id][t1.id] = d

    def _flush_overlaps(self):
        """Recompute overlap values for all trees marked dirty.

        Called at the start of competition methods. For each dirty tree,
        recomputes overlap_area with all its neighbors and updates both
        sides of the symmetric overlap caches.
        """

        for tid in self._dirty_ids:
            if tid not in self._nbr_dist:
                continue
            tree = self._id_to_tree[tid]
            ri_root = self.rootR[tree.size_ix]
            ri_canopy = self.canopyR[tree.size_ix]

            for nbr_id, d in self._nbr_dist[tid].items():
                nbr = self._id_to_tree[nbr_id]

                # root overlap
                if self.coeffs['area competition']>0:
                    a_root = overlap_area(d, ri_root, self.rootR[nbr.size_ix])
                    self._root_overlap[tid][nbr_id] = a_root
                    self._root_overlap[nbr_id][tid] = a_root

                # canopy overlap
                if self.coeffs['light competition']>0:
                    a_canopy = overlap_area(d, ri_canopy, self.canopyR[nbr.size_ix])
                    self._canopy_overlap[tid][nbr_id] = a_canopy
                    self._canopy_overlap[nbr_id][tid] = a_canopy

        self._dirty_ids.clear()

    def grow(self, dt):
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
                    self._dirty_ids.add(tree.id)
                else:
                    warn("Largest tree has reached max bin. Recommend increasing size range.")

        # introduce saplings
        for i in range(self.rng.poisson(self.g0 * dt)):
            new_id = next(self._next_id)
            self._next_id_value = new_id + 1
            new_tree = Tree(self.rng.uniform(0, self.L, size=2), self.t, id=new_id)
            self._nbr_dist_add(new_tree)
            self.trees.append(new_tree)
            self._id_to_tree[new_tree.id] = new_tree

        self.t += dt

    def kill(self, dt=1):
        """Kill trees across all size classes for one time step.

        Parameters
        ----------
        dt : float, 1.
            Time step.
        **kwargs
        """
        r = self.rng.rand(len(self.trees))
        killedTreeIx = []

        for i, tree in enumerate(self.trees):
            if r[i] < (self.deathRate[tree.size_ix] * dt):
                killedTreeIx.append(i)

        for ix in reversed(killedTreeIx):
            tree = self.trees.pop(ix)
            self._nbr_dist_remove(tree.id)
            self.deadTrees.append(tree.kill(self.t))

    def compete_area(self, dt=1):
        """Play out root area competition between trees to kill trees.

        Parameters
        ----------
        dt : float, 1.
            Time step.
        """
        if len(self.trees) < 2:
            return

        self._flush_overlaps()

        # randomly kill trees depending on whether or not below total basal met rate
        killedTreeIx = []
        xi = self.env_rng.rvs()  # current env status
        deathRate = self.coeffs['dep death rate'] * self.coeffs['area competition'] * dt

        for i, tree in enumerate(self.trees):
            ri = self.rootR[tree.size_ix]
            area_i = np.pi * ri**2

            # as an indpt pair approx just sum over all overlapping areas
            # to be precise, one should consider areas where multiple trees overlap as
            # different, but these correspond to high order interactions
            overlap_sum = sum(self._root_overlap[tree.id].values())

            dresource = (area_i - overlap_sum *
                         self.coeffs['sharing fraction']) * self.coeffs['resource efficiency']
            if ((self.basalMetRate[tree.size_ix] > (dresource / xi)) and (self.rng.rand() < deathRate)):
                killedTreeIx.append(i)

        for ix in reversed(killedTreeIx):
            tree = self.trees.pop(ix)
            self._nbr_dist_remove(tree.id)
            self.deadTrees.append(tree.kill(self.t))

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
        if len(self.trees) < 2:
            return

        self._flush_overlaps()

        rate = self.coeffs['light competition'] * dt

        # randomly kill trees with rate proportional to overlap and height diff
        killedTreeIx = []
        for i, tree in enumerate(self.trees):
            hi = self.canopyH[tree.size_ix]

            compete_factor = 0.
            for nbr_id, a in self._canopy_overlap[tree.id].items():
                if a > 0:
                    nbr = self._id_to_tree[nbr_id]
                    dh = self.canopyH[nbr.size_ix] - hi
                    compete_factor += a * rate * self.ldecay_f(dh)

            if run_checks and compete_factor > self.tol:
                warn("Competition rate could exceed rate tolerance limit. Recommend shrinking dt.")

            if self.rng.rand() < compete_factor:
                killedTreeIx.append(i)

        for ix in reversed(killedTreeIx):
            tree = self.trees.pop(ix)
            self._nbr_dist_remove(tree.id)
            self.deadTrees.append(tree.kill(self.t))
 
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
               iprint=False,
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
        iprint : bool, False
            If True, print progress information.
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
        timer = perf_counter()
        if n_forests==1:
            t = np.zeros(n_sample)
            nk = np.zeros((n_sample, self.kmax+1))
            trees = []
            total_iters = max(1, int(np.ceil(n_sample * sample_dt / dt)))
            last_bar_step = -1

            i = 0
            counter = 1  # for no. of samples saved, skipping initial condition
            while counter <= n_sample:
                # measure every dt, but make sure to account for potential floating point
                # precision errors
                if (i - counter * sample_dt / dt + 1e-15)>=0:
                    t[counter-1] = dt * i
                    nk[counter-1] = self.nk()
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

                if iprint:
                    bar_step = min(40, int(40 * min(i + 1, total_iters) / total_iters))
                    if bar_step != last_bar_step:
                        last_bar_step = bar_step
                        elapsed = perf_counter() - timer
                        pct = 100 * min(i + 1, total_iters) / total_iters
                        bar = '#' * bar_step + '.' * (40 - bar_step)
                        print(f"\rProgress [{bar}] {pct:6.2f}% ({elapsed:.1f}s)", end='', flush=True)
                i += 1

            if iprint:
                print()
            
            if return_trees:
                return nk, t, self.rRange, trees
            return nk, t, self.rRange

        # extract plain-Python params so the closure doesn't capture self
        # (self contains numba jitclass objects that can't be pickled)
        _L, _g0, _rRange = self.L, self.g0, self.rRange
        _coeffs, _nu, _bc = self.coeffs, self.nu, self.bc

        def loop_wrapper(args):
            forest = Forest2D(_L, _g0, _rRange, _coeffs, _nu, bc=_bc)
            if return_trees:
                return forest.sample(n_sample, dt, sample_dt, return_trees=True, **kwargs)
            return forest.sample(n_sample, dt, sample_dt, **kwargs)

        with threadpool_limits(limits=1, user_api='blas'):
            with Pool() as pool:
                if return_trees:
                    nk, t, rk, trees = list(zip(*pool.map(loop_wrapper, range(n_forests))))
                else:
                    nk, t, rk = list(zip(*pool.map(loop_wrapper, range(n_forests))))

        if return_trees:
            return nk, t, rk, trees
        return nk, t, rk

    def snapshot(self):
        """Return copy of self.trees."""

        return [_Tree.from_tree_nb(tree) for tree in self.trees]

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


class _Tree():
    """Tree container.

    Uses sentinel values instead of Python None (`id=-1`, `t=np.nan`).
    """

    def __init__(self, xy, t0=0.0, id=-1, size_ix=0, t=np.nan):
        self.xy = xy
        self.t0 = t0
        self.id = id
        self.size_ix = size_ix
        self.t = t

    def grow(self):
        self.size_ix += 1

    def kill(self, t):
        self.t = t
        return self

    def copy(self):
        return Tree(self.xy.copy(), self.t0, self.id, self.size_ix, self.t)

    def to_tree_nb(self):
        return Tree(self.xy, self.t0, self.id, self.size_ix, self.t)

    @classmethod 
    def from_tree_nb(cls, tree_nb):
        return _Tree(tree_nb.xy, tree_nb.t0, tree_nb.id, tree_nb.size_ix, tree_nb.t)

tree_nb_spec = [
    ('xy', types.float64[:]),
    ('t0', types.float64),
    ('id', types.int64),
    ('size_ix', types.int64),
    ('t', types.float64),
]


@jitclass(tree_nb_spec)
class Tree():
    """Numba-compatible tree container.

    Uses sentinel values instead of Python None (`id=-1`, `t=np.nan`).
    """

    def __init__(self, xy, t0=0.0, id=-1, size_ix=0, t=np.nan):
        self.xy = xy
        self.t0 = t0
        self.id = id
        self.size_ix = size_ix
        self.t = t

    def grow(self):
        self.size_ix += 1

    def kill(self, t):
        self.t = t
        return self

    def copy(self):
        return Tree(self.xy.copy(), self.t0, self.id, self.size_ix, self.t)
#end Tree

TREE_NB_TYPE = Tree.class_type.instance_type


def make_numba_tree_dict(trees):
    """Build an id->Tree typed dict suitable for passing into njit functions."""

    d = Dict.empty(key_type=types.int64, value_type=TREE_NB_TYPE)
    for tree in trees:
        if tree.id >= 0:
            d[int(tree.id)] = tree
    return d



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

@njit
def sum_overlap_area(nbr_dist_dict, _id_to_tree, rootR, ri):
    overlap_sum = 0.
    for nbr_id, d in nbr_dist_dict.items():
        nbr = _id_to_tree[nbr_id]
        rj = rootR[nbr.size_ix]
        overlap_sum += overlap_area(d, ri, rj)
    return overlap_sum

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

@njit
def incrementor(i0):
    i = i0-1
    while True:
        i += 1
        yield i

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

