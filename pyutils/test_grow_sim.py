# ====================================================================================== #
# Automata compartment model for forest growth.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .grow_sim import *



def test_overlap_area():
    assert overlap_area(2, 1, 1)==0
    assert overlap_area(2, 1, 1)==0
    assert overlap_area(0, 1, 1)==np.pi

def test_delete_flat_dist_rowcol():
    np.random.seed(0)
    n = 10

    d = np.random.rand(n * (n-1) // 2)
    dsquare = squareform(d)
    
    for i in range(n):
        newd = delete_flat_dist_rowcol(d, i, n)
        newdsquare = np.delete(np.delete(dsquare, i, axis=0), i, axis=1)
        assert np.array_equal(squareform(newd), newdsquare)

def test_append_flat_dist_rowcol(n=10):
    np.random.seed(0)

    d = np.random.rand(n * (n-1) // 2)
    print(d)
    newd = delete_flat_dist_rowcol(append_flat_dist_rowcol(d, -1, n), n, n+1)
    print(newd)
    assert np.array_equal(d, newd)
