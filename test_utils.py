# ====================================================================================== #
# Forest test utils.py
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *
from itertools import combinations
from scipy.integrate import quad



def test_row_ix_from_utri():
    for n in range(2, 6):
        pairs = list(combinations(range(n), 2))
        for i in range(n):
            ix = row_ix_from_utri(i, n)
        for ix_ in ix:
            assert i in pairs[ix_], (i, pairs[ix_])

