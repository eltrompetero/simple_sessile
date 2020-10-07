# ====================================================================================== #
# Forest test nearest_neighbor.py
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .nearest_neighbor import *
from itertools import combinations
from scipy.integrate import quad



def test_p():
    assert np.isclose(quad(p(1000, 10), 0, np.inf)[0], 1)
    assert np.isclose(quad(p(1000, 10, True), 0, np.inf)[0], 1)
