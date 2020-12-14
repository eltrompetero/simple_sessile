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

def mft_cutoff(nu=2.5, nForests=30):
    """Simulation of symmetric competition with changing cutoff modulated by adjusting
    basal metabolic rate coefficient.

    Parameters
    ----------
    nu : float, 2.5
    nForests : int, 30
        Number of indpt forests to run.
    """

    basalRange = np.array([.8, .2, .05, .0125, .003125])

    # set up
    r0 = 1  # smallest stem radius
    Abar = .5  # death rate coeff
    cg = .3  # growth rate coeff

    rRange = np.linspace(r0, 400, 2000)
    g0 = 500  # sapling introduction rate
    L = 200  # length of boundary
    sampleSize = 5000  # number of samples to take
    dt = .1  # sim time step

    nk = {}
    t = {}
    rk = {}

    for basal in basalRange:
        forest = Forest2D(L, g0, rRange, 
                          {'root':1,
                           'canopy':1,
                           'grow':cg,
                           'death':Abar,
                           'area competition':1,
                           'basal':basal,
                           'sharing fraction':.5,
                           'resource efficiency':2,
                           'dep death rate':1},
                          nu=nu)
        forest.check_dt(dt)

        nk[basal], t[basal], rk[basal] = forest.sample(sampleSize, dt,
                                                       sample_dt=.25,
                                                       n_forests=nForests)
        save_pickle(['nk','t','rk','forest','r0','g0','nu','Abar','basalRange','cg'],
                    f'cache/biomass_scaling_w_compet_{nu=}.p', True)
        print(f'Done with {basal=}.')

def mft_cutoff_finite_size_checks(nu=2.5, run_smaller=True, run_larger=True):
    """Simulation of symmetric competition with changing cutoff modulated by adjusting
    basal metabolic rate coefficient. This decreases and increases area generated by
    .mft_cutoff() by a factor of 4 to test for finite size effects.

    Parameters
    ----------
    nu : float, 2.5
    run_smaller : bool, True
    run_larger : bool, True
    """

    basalRange = np.array([.8, .2, .05, .0125, .003125])

    # set up
    r0 = 1
    Abar = .5
    cg = .3

    rRange = np.linspace(r0, 400, 2000)
    sampleSize = 5000
    dt = .1
    nForests = 30

    # smaller system
    if run_smaller:
        g0 = 500 / 4
        L = 200 / 2
        nk = {}
        t = {}
        rk = {}

        for basal in basalRange:
            forest = Forest2D(L, g0, rRange, 
                              {'root':1,
                               'canopy':1,
                               'grow':cg,
                               'death':Abar,
                               'area competition':1,
                               'basal':basal,
                               'sharing fraction':.5,
                               'resource efficiency':2,
                               'dep death rate':1},
                              nu=nu)
            forest.check_dt(dt)

            nk[basal], t[basal], rk[basal] = forest.sample(sampleSize, dt,
                                                           sample_dt=.25,
                                                           n_forests=nForests)
            
            save_pickle(['nk','t','rk','forest','r0','g0','nu','Abar','basalRange','cg'],
                        f'cache/biomass_scaling_w_compet_smaller_{nu=}.p', True)
            print(f'Done with {basal=}.')
    
    # larger system
    if run_larger:
        g0 = 500 * 4
        L = 200 * 2

        nk = {}
        t = {}
        rk = {}

        for basal in basalRange:
            forest = Forest2D(L, g0, rRange, 
                              {'root':1,
                               'canopy':1,
                               'grow':cg,
                               'death':Abar,
                               'area competition':1,
                               'basal':basal,
                               'sharing fraction':.5,
                               'resource efficiency':2,
                               'dep death rate':1},
                              nu=nu)
            forest.check_dt(dt)

            nk[basal], t[basal], rk[basal] = forest.sample(sampleSize, dt,
                                                           sample_dt=.25,
                                                           n_forests=nForests)
            
            save_pickle(['nk','t','rk','forest','r0','g0','nu','Abar','basalRange','cg'],
                        f'cache/biomass_scaling_w_compet_larger_{nu=}.p', True)
            print(f'Done with {basal=}.')

def mft_cutoff_plot(nu=2.5):
    """Forest examples for showing individual plots.

    Parameters
    ----------
    nu : float, 2.5
    """

    basalRange = np.array([.8, .2, .05, .0125, .003125])

    # set up
    r0 = 1  # smallest stem radius
    Abar = .5  # death rate coeff
    cg = .3  # growth rate coeff

    rRange = np.linspace(r0, 400, 2000)
    g0 = 500  # sapling introduction rate
    L = 200  # length of boundary
    sampleSize = 4000  # number of samples to take
    dt = .1  # sim time step

    def loop_wrapper(basal):
        forest = Forest2D(L, g0, rRange, 
                          {'root':1,
                           'canopy':1,
                           'grow':cg,
                           'death':Abar,
                           'area competition':1,
                           'basal':basal,
                           'sharing fraction':.5,
                           'resource efficiency':2,
                           'dep death rate':1},
                          nu=nu)
        forest.check_dt(dt)

        forest.sample(sampleSize, dt,
                      sample_dt=.25);
        return forest

    with Pool(basalRange.size) as pool:
        forest = dict(zip(basalRange, pool.map(loop_wrapper, basalRange)))

    save_pickle(['forest'],
                f'plotting/biomass_scaling_w_compet_{nu=}.p', True)

def phase_space_scan_Abar():
    """Scanning across natural mortality rate Abar as in Figure 3.
    """

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
    sampleSize = 100
    dt = .1
    coeffs = {'root':10,
              'canopy':1,
              'grow':cg,
              'area competition':1,
              'basal':basal,
              'sharing fraction':1,
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
            
            print(f'Done with {deathRate=:.3f}.')
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

def phase_space_scan_abar():
    """Scanning across varying growth rate fixing natural mortality rate to 0 as in Figure
    3.
    """

    # for showing the spatial distributions
    cgRange = np.logspace(np.log10(.5), -4, 4)
    areaDeathRateRange = np.logspace(-1, 2, 10)  # keys to dicts in xy

    # set up
    r0 = 1
    Abar = 0.
    basal = .05

    rRange = np.linspace(r0, 400, 800)  # growth saturates 
    g0 = 100
    L = 200
    burnIn = 400
    sampleSize = 100
    dt = .1
    coeffs = {'root':10,
              'death':Abar,
              'area competition':1,
              'basal':basal,
              'sharing fraction':1,
              'resource efficiency':2}

    def loop_cg(cg):
        coeffs['grow'] = cg

        def loop_wrapper(deathRate):
            coeffs['dep death rate'] = deathRate
            forest = Forest2D(L, g0, rRange, coeffs)
            forest.check_dt(dt)

            # burn in and run sim
            if deathRate>1:
                forest.sample(2, dt=dt, sample_dt=burnIn+1600)
            else:
                forest.sample(2, dt=dt, sample_dt=burnIn)
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
    for cg in cgRange:
        xy[cg], nk[cg] = loop_cg(cg)
        save_pickle(['cgRange','areaDeathRateRange','r0','cg','basal','rRange',
                     'g0','L','burnIn','sampleSize','dt','coeffs','xy','nk'],
                    f'cache/spacing_with_cg.p', True)
        print(f'Done with {cg=}.')
        print('')

def hex_packing():
    """Hexagonal packing emerging from strong rate competition. The results from this can
    be used to generate Figure 4D.
    """

    from .nearest_neighbor import pair_correlation

    # for showing the spatial distributions
    areaDeathRateRange = np.logspace(-1, 3, 10)  # keys to dicts in xy

    # set up
    r0 = 1
    basal = 0

    rRange = np.linspace(r0, 5, 5)  # growth saturates b/c max radius is 5
    g0 = 100
    L = 200
    burnIn = 1_000  # in time steps
    sampleSize = 1_000
    dt = .2
    coeffs = {'root':10,
              'death':0,
              'grow':.3,
              'area competition':1,
              'basal':basal,
              'sharing fraction':1,
              'resource efficiency':2}

    def loop_wrapper(deathRate):
        coeffs['dep death rate'] = deathRate
        forest = Forest2D(L, g0, rRange, coeffs)
        forest.check_dt(dt)

        # burn in and run sim
        forest.sample(2, dt=dt, sample_dt=burnIn)
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

    save_pickle(['areaDeathRateRange','r0','basal','rRange',
                 'g0','L','burnIn','sampleSize','dt','coeffs','xy','nk'],
                'cache/packing_example.p', True)
    
    # for plotting the correlation fcn
    allxy = xy
    p = {}
    bins = np.linspace(0, 5, 40)  # this should be roughly aligned with the stats of the system
    for adr in areaDeathRateRange:
        # fix natural mortality and titrate strength of competition
        xy = allxy[adr]

        # iterate through each random plot
        thisp = []
        r = []
        for xy_ in xy:
            p_, r_ = pair_correlation(np.vstack(xy_), bins, (50, 50, 100, 100))
            thisp.append(p_)
            r.append(r_)

        p[adr] = np.vstack(thisp).mean(0)
        
    r = r[0]  # x-axis, radial distance

    save_pickle(['p','r'], 'plotting/spatial_correlation.p')
