# Merge per-pair pickles from rainfall_posterior_job.py into the single
# cache/posterior_rainfall.p that 'BCI rainfall 2024-02-21.ipynb' expects.
# The saved dict {'params', 'x0', 'ecdf'} matches workspace.utils.load_pickle,
# so `load_pickle('cache/posterior_rainfall.p')` defines all three variables.
import os
import pickle

params = {}
x0 = {}
ecdf = {}
missing = []
for census_year in [1990, 1995, 2000, 2005, 2010, 2015, 2020]:
    for kernel_duration in range(2, 15):
        f = f'cache/posterior_rainfall/{census_year}_{kernel_duration}.p'
        if not os.path.isfile(f):
            missing.append((census_year, kernel_duration))
            continue
        with open(f, 'rb') as fp:
            d = pickle.load(fp)
        params[(census_year, kernel_duration)] = d['params']
        x0[(census_year, kernel_duration)] = d['x0']
        ecdf[(census_year, kernel_duration)] = d['ecdf']

if missing:
    print(f'WARNING: {len(missing)} pairs missing, not merging: {missing}')
else:
    with open('cache/posterior_rainfall.p', 'wb') as fp:
        pickle.dump({'params':params, 'x0':x0, 'ecdf':ecdf}, fp, -1)
    print(f'wrote cache/posterior_rainfall.p with {len(params)} pairs')
