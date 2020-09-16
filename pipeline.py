# ====================================================================================== #
# Pipeline tools for analysis.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *
from .grow_sim import Forest2D
import pandas as pd



def WEB_transience():
    # Set up common parameters
    g0 = 1000
    L = 10
    nSample = 200
    nForests = 40

    cm = .5
    cg = .3
    dt = .005

    # Thin bins to show alignment between prediction and theory.
    # set up
    rRange = np.linspace(1, 500, 5000)
    forest = Forest2D(L, g0, rRange,
		      {'root':1,
		       'grow':cg,
		       'death':cm})

    nk, t, rk = forest.sample(nSample, dt, n_forests=nForests)

    save_pickle(['rRange','g0','L','nSample','cm','cg','dt','t','nk','rk','forest'],
		'cache/linear_model_exponent_transience.p', True)

    # Thick bins to show deviations at small r.
    # set up
    rRange = np.linspace(1, 500, 500)
    forest = Forest2D(L, g0, rRange,
		      {'root':1,
		       'grow':cg,
		       'death':cm})

    nk, t, rk = forest.sample(nSample, dt, n_forests=nForests)

    save_pickle(['rRange','g0','L','nSample','cm','cg','dt','t','nk','rk','forest'],
		'cache/linear_model_exponent_transience_wide_bins.p', True)


