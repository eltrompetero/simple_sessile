# ====================================================================================== #
# Automata compartment model for forest growth.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .grow_sim import *



def test_overlap_area():
    assert overlap_area((0,0), 1, (2,0), 1)==0
    assert overlap_area((0,0), 1, (0,2), 1)==0
    assert overlap_area((0,0), 1, (0,0), 1)==np.pi
