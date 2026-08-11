# Merge the eight full-data LKW posteriors
# (cache/large_kappa/prior_sensitivity/{census}_full.p, from lkw_prior_job.py /
# lkw_full.sbatch) into cache/posterior_lkw_full.p with the same structure as
# cache/posterior_lkw.p: {'params': {census: {'alpha','kappa','rstar','b'}}, 'meta': ...}.
# Unlike the original, there is no 'potential_energy' (the job script does not
# store it) and the data are the full alive dbh>=20 sample, not an 80k subsample.
import os
import pickle

import numpy as np

params = {}
meta = {}
missing = []
for census in range(1, 9):
    f = f'cache/large_kappa/prior_sensitivity/{census}_full.p'
    if not os.path.isfile(f):
        missing.append(census)
        continue
    with open(f, 'rb') as fp:
        d = pickle.load(fp)
    params[census] = d['params']
    meta[census] = d['meta']

if missing:
    print(f'WARNING: censuses missing, not merging: {missing}')
else:
    with open('cache/posterior_lkw_full.p', 'wb') as fp:
        pickle.dump({'params': params, 'meta': meta}, fp, -1)
    print(f'wrote cache/posterior_lkw_full.p with {len(params)} censuses')
    for c in range(1, 9):
        k = params[c]['kappa']
        a = params[c]['alpha']
        r = params[c]['rstar']
        print(f'census {c}: alpha {a.mean():.3f}±{a.std():.3f}  '
              f'kappa {k.mean():.3f}±{k.std():.3f}  rstar {r.mean():.0f}±{r.std():.0f}  '
              f'(n_data {meta[c]["n_data"]})')
