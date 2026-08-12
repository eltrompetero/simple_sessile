# Sample the rainfall posterior for a single (census_year, kernel_duration) pair.
# Extracted from 'BCI rainfall 2024-02-21.ipynb' so each pair runs in its own
# process (fresh JAX compilation cache) instead of one ever-growing kernel.
#
# Run from ./py:  python rainfall_posterior_job.py CENSUS_YEAR KERNEL_DURATION [--smoke]
# Writes cache/posterior_rainfall/{CENSUS_YEAR}_{KERNEL_DURATION}.p and skips
# pairs whose output already exists, so a partial sweep can be resubmitted.
import os
import sys
import pickle
import warnings

from pyutils.rainfall import *
from pyutils.posterior import *


def detect_lower_cutoff(x, min_threshold_factor_range, initial_avg_n=3, pct_change_threshold=.3):
    """Use a percent-slope-change threshold method to determine lower cutoff for power law fit.

    See 'BCI rainfall 2024-02-21.ipynb' for discussion.

    Parameters
    ----------
    x : ndarray
        Samples from distribution to model with power law.
    min_threshold_factor_range : ndarray
        Coefficients with which to multiple the min value in x to determine cutoff.
    initial_avg_n : int, 3
    pct_change_threshold : float, .3

    Returns
    -------
    float
        Lower threshold factor to use. Upon failure, it returns the minimum factor given.
    """
    assert x.min()>0
    assert min_threshold_factor_range.min()>=1

    alpha_naive = np.zeros_like(min_threshold_factor_range)
    for i, min_threshold_factor in enumerate(min_threshold_factor_range):
        x0 = x.min() * min_threshold_factor
        alpha_naive[i] = 1/np.log(x[x>=x0]/x0).mean()

    dalpha = np.abs(np.diff(alpha_naive)/alpha_naive[:-1])
    initial_change = dalpha[:initial_avg_n].mean()
    try:
        ix = np.where(dalpha<(initial_change*pct_change_threshold))[0][0] + 1
    except IndexError:
        ix = 0

    return min_threshold_factor_range[ix]


def sample_posterior(census_year, kernel_duration=7,
                     key=random.PRNGKey(43),
                     iprint=False,
                     mcmc_settings={}):
    """Sample posterior distribution for rainfall parameters for the specific census
    kernel duration.

    Unlike the notebook version, the census interval is derived from census_year
    instead of relying on a global `years`. The census year is the last year of
    the interval: years = (census_year-6, census_year+1), range() convention.

    Parameters
    ----------
    census_year : int
    kernel_duration : int, 7
    key : jax.random.PRNGKey
    iprint : bool, False
    mcmc_settings : dict
        Overrides for the MCMC keyword arguments (used by --smoke).

    Returns
    -------
    dict
        Posterior samples for power-law parameters.
    float
        Lower cutoff for power law fit.
    statsmodels.distributions.empirical_distribution.ECDF
        ECDF of the inverse rainfall data.
    """
    # check parameters
    assert census_year in [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
    if census_year in [1980, 1985]:
        warnings.warn('The first two surveys (1980, 1985) have badly rounded data.')
    years = (census_year - 6, census_year + 1)
    min_threshold_factor_range = np.logspace(0, 1.5, 35)
    assert 2<=kernel_duration<=14

    # rainfall preprocessing; smoothing is now exact in log space, so smoothed_ra
    # is finite and positive for every sample after the record's first rain and no
    # upper truncation of ira is needed -- ra>0 only guards the rare exact zeros
    bci_rainfall.smooth_rainfall_exp(kernel_duration)
    df = bci_rainfall.by_year(years)
    ra = df['smoothed_ra'].values.ravel()
    ecdf = ECDF(1/ra[ra>0])

    # inverse rainfall
    ira = 1/ra[ra>0]
    lc_f = detect_lower_cutoff(ira, min_threshold_factor_range)
    x0 = ira.min() * lc_f
    ira = ira[ira>=x0]

    def tpl(X, x0=x0,
            alpha_mean=0., alpha_std=1.,
            el_mean=-5., el_std=1.,
            num_samples=100):
        alpha = pyro.sample("alpha", Uniform(low=.01, high=3))
        el = pyro.sample("el", LogNormal(el_mean, el_std))

        with pyro.plate("data", X.shape[0] if X is not None else num_samples):
            pyro.sample("trees", ExpTruncatedPowerLaw(alpha, el, x0=jnp.array([x0*1.])), obs=X)

    mcmc_kwargs = dict(num_warmup=1_000,
                       num_samples=2_000,
                       num_chains=16,
                       thinning=20,
                       progress_bar=False)
    mcmc_kwargs.update(mcmc_settings)

    # fit the full sample: the deep-drought tail holds only tens of samples, so a
    # 40k subsample sees it by luck; the key now seeds the chains (reproducible)
    sample = collect_rainfall_sample(tpl, jnp.array(ira), 1,
                                     key=key,
                                     iprint=iprint,
                                     **mcmc_kwargs)

    params = consolidate_sample(sample)

    return params, x0, ecdf


if __name__ == '__main__':
    census_year = int(sys.argv[1])
    kernel_duration = int(sys.argv[2])
    smoke = '--smoke' in sys.argv[3:]

    outdr = 'cache/posterior_rainfall'
    os.makedirs(outdr, exist_ok=True)
    outfile = f'{outdr}/{census_year}_{kernel_duration}.p' if not smoke else f'{outdr}/smoke.p'
    if os.path.isfile(outfile) and not smoke:
        print(f'{outfile} already exists; skipping.')
        sys.exit(0)

    mcmc_settings = {} if not smoke else dict(num_warmup=100, num_samples=100,
                                              num_chains=2, thinning=2)

    bci_rainfall = BCI_Rainfall()
    params, x0, ecdf = sample_posterior(census_year, kernel_duration,
                                        iprint=True,
                                        mcmc_settings=mcmc_settings)

    with open(outfile, 'wb') as f:
        pickle.dump({'params':params, 'x0':x0, 'ecdf':ecdf}, f, -1)
    print(f'wrote {outfile}')
