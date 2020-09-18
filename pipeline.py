# ====================================================================================== #
# Pipeline tools for analysis.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *
from .grow_sim import Forest2D
import pandas as pd
from workspace.utils import save_pickle



def WEB_transience():
    """Show moving cutoff once starting from an empty plot for the simple WEB compartment
    model.
    """

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

def phase_space_scan():
    # Scanning across natural mortality rate Abar.
    AbarRange = np.linspace(.75, 0, 5)  # keys to xy dict
    areaDeathRateRange = np.logspace(-1, 2, 10)  # keys to dicts in xy

    # set up
    r0 = 1
    cg = .3
    nu = 2.
    basal = .05

    rRange = np.linspace(r0, 800, 1600)  # growth saturates 
    g0 = 100
    L = 200
    burnIn = 400
    sampleSize = 1_000
    dt = .1
    coeffs = {'root':10,
              'canopy':1,
              'grow':cg,
              'area competition':1,
              'basal':basal,
              'sharing fraction':.5,
              'resource efficiency':2}

    def loop_Abar(Abar):
        coeffs['death'] = Abar

        def loop_wrapper(deathRate):
            coeffs['dep death rate'] = deathRate
            forest = Forest2D(L, g0, rRange, coeffs,
                              nu=nu)
            forest.check_dt(dt)

            # burn in and run sim
            if Abar<.38 and deathRate>1:  # long time to converge in this regime
                if Abar<.2:
                    forest.sample(burnIn+1000, dt=dt, sample_dt=sampleSize * dt)
                else:
                    forest.sample(burnIn+400, dt=dt, sample_dt=sampleSize * dt)
            else:
                forest.sample(burnIn, dt=dt, sample_dt=sampleSize * dt)
            nk, t, rk, trees = forest.sample(sampleSize, dt=dt, sample_dt=10, return_trees=True)
            
            # get tree coordinates
            xy = [np.vstack([tree.xy for tree in thisTrees]) for thisTrees in trees]
            
            print(f'Done with {deathRate=:.2f}.')
            return xy, nk

        with threadpool_limits(user_api='blas', limits=1):
            with Pool(cpu_count()-1) as pool:
                xy_, nk_ = list(zip(*pool.map(loop_wrapper, areaDeathRateRange)))
                xy = dict(zip(areaDeathRateRange, xy_))
                nk = dict(zip(areaDeathRateRange, nk_))
                
        return xy, nk

    xy = {}  # loop over mortality rates
    nk = {}  # pop. number (can be used for equilibrium check)
    for Abar in AbarRange:
        xy[Abar], nk[Abar] = loop_Abar(Abar)
        save_pickle(['AbarRange','areaDeathRateRange','r0','cg','nu','basal','rRange',
                     'g0','L','burnIn','sampleSize','dt','coeffs','xy','nk'],
                    'cache/phase_space_scan_Abar.p', True)
        print(f'Done with {Abar=}.')
        print('')
