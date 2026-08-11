# Prior/procedure sensitivity sweep for the LKW tree-size posterior of
# 'posterior sampling 2025-04-11.ipynb' (cells 32/34), targeting the elevated
# census-6 kappa. Each (census, variant) runs in its own process.
#
# Run from ./py:  python lkw_prior_job.py CENSUS VARIANT [--smoke]
# Writes cache/large_kappa/prior_sensitivity/{CENSUS}_{VARIANT}.p and skips
# outputs that already exist, so a partial sweep can be resubmitted.
#
# Variants (seed = PRNGKey argument):
#   seedA/seedB/seedC : original priors, 80k with-replacement subsample,
#                       seeds 101/202/303 (seed varies BOTH subsample and chains)
#   full              : original priors, full census sample (no subsampling)
#   kprior1           : kappa ~ LogNormal(0, 1)     (prior median 1 vs 0.05)
#   kprior2           : kappa ~ LogNormal(-3, 0.5)  (tight small-kappa stress test)
#   rprior            : rstar ~ LogNormal(6.4, 1)   (centered near data scale ~600mm)
#   deeptree          : original priors, NUTS max_tree_depth=10 (vs 6 in collect_sample)
#
# Randomness matches the original procedure: key = PRNGKey(seed);
# key, subkey = random.split(key); subsample with subkey; pass key into the MCMC.
import os
import sys
import pickle
from functools import partial

from pyutils.posterior import *  # pyro, jnp, np, random, Uniform, LogNormal,
                                 # NUTS, MCMC, init_to_sample, MeanFieldLKW,
                                 # collect_sample, consolidate_sample


def lkw_mft(X, r0=20, b=1/3,
            kappa_mean=-3., kappa_std=2.,
            rstar_mean=10., rstar_std=1,
            num_samples=100):
    """LKW mean-field tree-size model, copied verbatim from cell 32 of
    'posterior sampling 2025-04-11.ipynb'."""
    alpha = pyro.sample("alpha", Uniform(low=.5, high=3))
    kappa = pyro.sample("kappa", LogNormal(kappa_mean, kappa_std))
    rstar = pyro.sample("rstar", LogNormal(rstar_mean, rstar_std))

    constraint = kappa + 1 - b
    pyro.factor("constraint", jnp.where(constraint > 0, 0.0, -jnp.inf))

    with pyro.plate("data", X.shape[0] if X is not None else num_samples):
        pyro.sample("spins", MeanFieldLKW(alpha, kappa, b, rstar, r0=jnp.array(r0*1.)), obs=X)


# defaults reproduce the original run; each variant overrides selected entries
DEFAULTS = dict(seed=101,
                subsample=True,
                kappa_mean=-3., kappa_std=2.,
                rstar_mean=10., rstar_std=1.,
                max_tree_depth=6)

VARIANTS = {
    'seedA':    dict(seed=101),
    'seedB':    dict(seed=202),
    'seedC':    dict(seed=303),
    'full':     dict(subsample=False),
    'kprior1':  dict(kappa_mean=0., kappa_std=1.),
    'kprior2':  dict(kappa_mean=-3., kappa_std=.5),
    'rprior':   dict(rstar_mean=6.4, rstar_std=1.),
    'deeptree': dict(max_tree_depth=10),
}

SUBSAMPLE_SIZE = 80_000


def load_dbh(census):
    """dbh of alive main stems with dbh>=20 mm for one census, as in the
    original pull. Falls back to the cached extraction if duckdb fails."""
    try:
        from pyutils.data import BCI
        bci = BCI()
        dbh = bci.execute(f"select dbh from bci where dataset_id={census} "
                          "and status='A' and dbh>=20").values.ravel()
    except Exception as e:
        print(f'BCI() failed ({e!r}); falling back to cache/large_kappa/dbh20.npz')
        dbh = np.load('cache/large_kappa/dbh20.npz')[str(census)]
    return jnp.array(dbh, dtype=jnp.float64)


def sample_posterior(census, variant, mcmc_settings={}):
    """Sample the LKW posterior for one census under one variant.

    Returns
    -------
    dict
        Consolidated posterior samples, keys 'alpha', 'kappa', 'rstar', 'b'.
    dict
        Metadata describing the variant actually run.
    """
    cfg = DEFAULTS | VARIANTS[variant]

    dbh = load_dbh(census)
    print(f'census {census}: {dbh.size} dbh values, min {dbh.min()}')

    # split exactly as the original loop did, so the subsample and the chains
    # both vary with the seed
    key = random.PRNGKey(cfg['seed'])
    key, subkey = random.split(key)
    if cfg['subsample']:
        X = random.choice(subkey, dbh, shape=(min(SUBSAMPLE_SIZE, dbh.size),))
    else:
        X = dbh

    model = partial(lkw_mft,
                    kappa_mean=cfg['kappa_mean'], kappa_std=cfg['kappa_std'],
                    rstar_mean=cfg['rstar_mean'], rstar_std=cfg['rstar_std'])

    mcmc_kwargs = dict(num_warmup=1_000,
                       num_samples=2_000,
                       num_chains=16,
                       thinning=20,
                       progress_bar=False)
    mcmc_kwargs.update(mcmc_settings)

    if cfg['max_tree_depth'] == 6:
        sample, V = collect_sample(model, X, 1, key=key, iprint=True, **mcmc_kwargs)
    else:
        # collect_sample hardcodes max_tree_depth=6, so mirror its body here
        # with the requested depth (n_loop=1)
        import time as time_
        t0 = time_.time()
        nuts_kernel = NUTS(model, dense_mass=True, max_tree_depth=cfg['max_tree_depth'],
                           init_strategy=init_to_sample)
        sampler = MCMC(nuts_kernel, **mcmc_kwargs)
        key, subkey = random.split(key)
        sampler.run(subkey, X, extra_fields=('potential_energy',))
        sample = [sampler.get_samples(True).copy()]
        print(f'Done with iteration 1/1 in {time_.time()-t0:.2f} seconds.')

    params = consolidate_sample(sample)
    params['b'] = np.zeros_like(params['alpha']) + 1/3

    meta = {'census': census,
            'variant': variant,
            'seed': cfg['seed'],
            'n_data': int(X.shape[0]),
            'subsample': cfg['subsample'],
            'priors': {'alpha': ('Uniform', .5, 3.),
                       'kappa': ('LogNormal', cfg['kappa_mean'], cfg['kappa_std']),
                       'rstar': ('LogNormal', cfg['rstar_mean'], cfg['rstar_std'])},
            'max_tree_depth': cfg['max_tree_depth'],
            'mcmc_kwargs': mcmc_kwargs}
    return params, meta


if __name__ == '__main__':
    census = int(sys.argv[1])
    variant = sys.argv[2]
    smoke = '--smoke' in sys.argv[3:]
    assert census in range(1, 9)
    assert variant in VARIANTS

    outdr = 'cache/large_kappa/prior_sensitivity'
    os.makedirs(outdr, exist_ok=True)
    outfile = f'{outdr}/{census}_{variant}.p' if not smoke else f'{outdr}/smoke.p'
    if os.path.isfile(outfile) and not smoke:
        print(f'{outfile} already exists; skipping.')
        sys.exit(0)

    mcmc_settings = {} if not smoke else dict(num_warmup=100, num_samples=100,
                                              num_chains=2, thinning=2)

    params, meta = sample_posterior(census, variant, mcmc_settings=mcmc_settings)
    for k in ['alpha', 'kappa', 'rstar']:
        print(f'<{k}> = {params[k].mean():.4f} +/- {params[k].std():.4f}  shape {params[k].shape}')

    with open(outfile, 'wb') as f:
        pickle.dump({'params': params, 'meta': meta}, f, -1)
    print(f'wrote {outfile}')
