# ====================================================================================== #
# Tests for automata compartment model for forest growth.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .grow_sim import *
from .grow_sim import _ghost_positions


def test_overlap_area():
    assert overlap_area(2, 1, 1)==0
    assert overlap_area(0, 1, 1)==np.pi
    # partial overlap: two identical circles offset by one radius
    a = overlap_area(1, 1, 1)
    assert 0 < a < np.pi

def test_pair_dist_euclidean():
    """With L=0, pair_dist should be standard Euclidean distance."""
    a = np.array([1., 2.])
    b = np.array([4., 6.])
    expected = np.sqrt(9 + 16)
    assert pair_dist(a, b, 0.) == expected

def test_pair_dist_interior_periodic_matches_free():
    """For interior points far from edges, periodic and free distances should be identical."""
    L = 100.
    a = np.array([40., 50.])
    b = np.array([45., 55.])
    assert pair_dist(a, b, 0.) == pair_dist(a, b, L)

def test_pair_dist_wrapping():
    """Test all wrapping cases for toroidal distance."""
    L = 100.

    # x-wrap only
    a = np.array([1., 50.])
    b = np.array([99., 50.])
    assert np.isclose(pair_dist(a, b, L), 2.)

    # y-wrap only
    a = np.array([50., 1.])
    b = np.array([50., 99.])
    assert np.isclose(pair_dist(a, b, L), 2.)

    # both wrap
    a = np.array([1., 1.])
    b = np.array([99., 99.])
    assert np.isclose(pair_dist(a, b, L), np.sqrt(8))

    # neither wraps (close together in interior)
    a = np.array([50., 50.])
    b = np.array([53., 54.])
    assert np.isclose(pair_dist(a, b, L), pair_dist(a, b, 0.))

def test_jit_overlap_area_free_vs_periodic_interior():
    """For trees far from edges, jit_overlap_area should give identical results
    regardless of boundary condition."""
    L = 1000.
    rng = np.random.RandomState(42)
    n = 20

    # place all trees in center, far from edges
    xy = rng.uniform(400, 600, size=(n, 2))
    r = rng.uniform(1, 5, size=n)

    overlap_free = jit_overlap_area(xy, r, 0.)
    overlap_periodic = jit_overlap_area(xy, r, L)

    assert np.allclose(overlap_free, overlap_periodic)

def test_jit_overlap_area_periodic_finds_boundary_neighbors():
    """Two trees near opposite edges should have zero overlap with free BC
    but nonzero overlap with periodic BC."""
    L = 100.
    xy = np.array([[1., 50.],
                   [99., 50.]])
    # radii large enough that toroidal distance (2.0) causes overlap
    r = np.array([3., 3.])

    overlap_free = jit_overlap_area(xy, r, 0.)
    overlap_periodic = jit_overlap_area(xy, r, L)

    assert overlap_free[0] == 0.
    assert overlap_periodic[0] > 0.

def test_jit_overlap_area_vs_sparse_overlap_area():
    """Per-tree overlap sums from the dense and sparse methods should agree
    for both free and periodic boundary conditions."""
    rng = np.random.RandomState(123)
    n = 30
    L = 50.

    xy = rng.uniform(0, L, size=(n, 2))
    r = rng.uniform(1, 5, size=n)

    for _L in [0., L]:
        dense = jit_overlap_area(xy, r, _L)
        overlap_sum_sparse, _ = sparse_overlap_area(xy, r, _L)

        # reconstruct per-tree sums from the dense condensed vector
        overlap_sum_dense = np.zeros(n)
        for i in range(n):
            overlap_sum_dense[i] = dense[row_ix_from_utri(i, n)].sum()

        assert np.allclose(overlap_sum_dense, overlap_sum_sparse), \
            f"Mismatch for L={_L}: max diff={np.max(np.abs(overlap_sum_dense - overlap_sum_sparse))}"

def test_ghost_positions_interior():
    """Interior tree should produce only its own position."""
    xy = np.array([50., 50.])
    positions = _ghost_positions(xy, 5., 100., 'periodic')
    assert len(positions) == 1

    # free BC always returns just one regardless of position
    xy_edge = np.array([1., 1.])
    positions = _ghost_positions(xy_edge, 5., 100., 'free')
    assert len(positions) == 1

def test_ghost_positions_edge():
    """Tree near one edge should produce 2 positions (original + ghost)."""
    # near left edge only
    xy = np.array([2., 50.])
    positions = _ghost_positions(xy, 5., 100., 'periodic')
    assert len(positions) == 2
    # ghost should be shifted +L in x
    assert np.isclose(positions[1][0], 102.)

def test_ghost_positions_corner():
    """Tree near a corner should produce 4 positions
    (original + x-ghost + y-ghost + diagonal ghost)."""
    xy = np.array([2., 2.])
    positions = _ghost_positions(xy, 5., 100., 'periodic')
    assert len(positions) == 4
