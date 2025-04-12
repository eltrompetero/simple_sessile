# Posterior sampling module using JAX and numpyro.
# Author: Eddie Lee, edlee@csh.ac.at
import numpyro as pyro
from numpyro.distributions import Uniform, LogNormal
from numpyro.infer import NUTS, MCMC
from numpyro.infer.initialization import init_to_sample
from numpyro.distributions import constraints
import jax.numpy as jnp
import jax.scipy as jsc
from jax import jit, random, lax
from jax.scipy.optimize import minimize
import jax
import time
import numpy as np
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
pyro.set_platform('cpu')
pyro.set_host_device_count(16)



# ================ #
# Helper functions #
# ================ #
def collect_sample(model,
                   X,
                   n_loop,
                   key=None,
                   iprint=True,
                   **mcmc_kwargs):
    """MCMC with NUTS kernel to obtain samples from posterior.

    Parameters
    ----------

    """
    sample = []
    key = key if not key is None else random.PRNGKey(np.random.randint(2**32-1))

    for i in range(n_loop):
        if iprint: t0 = time.time()
        nuts_kernel = NUTS(model, dense_mass=True, max_tree_depth=6, init_strategy=init_to_sample)
        sampler = MCMC(nuts_kernel, **mcmc_kwargs)
        
        key, subkey = random.split(key)
        sampler.run(subkey, X)
        sample.append(sampler.get_samples().copy())
        if iprint: print(f"Done with iteration {i+1}/{n_loop} in {time.time()-t0:.2f} seconds.")
    return sample

def consolidate_sample(sample):
    """Organize list of samples from collect_sample().
    """
    params = dict([(k,[]) for k in sample[0].keys()])
    for s in sample:
        for k in s.keys():
            params[k] += s[k].tolist()
    for k in params.keys():
        params[k] = np.array(params[k])
    return params

def describe_params(params):
    assert isinstance(params, dict)
    for k in params.keys():
        print(f'<{k}> = {params[k].mean()} +/- {params[k].std()}')

def cdf_bounds(xplot, params, percentile=(5,95)):
    """Return CDF with confidence bounds given the parameter samples along
    the given x-coordinates. This is a helper function for plotting.
    
    Parameters
    ----------
    xplot : jnp.array
    params : dict
        Keys 'alpha', 'kappa', 'b', and 'rstar' that give list-like objects.

    Returns
    -------
    np.ndarray
    """
    n_sample = params[list(params.keys())[0]].size
    cdf = jnp.zeros(xplot.size)
    cdf_min = jnp.zeros(xplot.size)
    cdf_max = jnp.zeros(xplot.size)

    def inner(j, val):
        x, params, cdf_ = val
        mft = MeanFieldLKW(params['alpha'][j],
                           params['kappa'][j],
                           params['b'][j],
                           params['rstar'][j],
                           r0=jnp.array(20.))
        return [x, params, cdf_.at[j].set(mft.cdf(jnp.array([x]))[0])]

    def outer(i, val):
        xplot, cdf, cdf_min, cdf_max = val
        x = xplot[i]

        cdf_ = lax.fori_loop(0, n_sample, inner, [x, params, jnp.zeros(n_sample)])[-1]

        cdf = cdf.at[i].set(jnp.median(cdf_))
        cdf_min = cdf_min.at[i].set(jnp.percentile(cdf_, percentile[0]))
        cdf_max = cdf_max.at[i].set(jnp.percentile(cdf_, percentile[1]))

        return [xplot, cdf, cdf_min, cdf_max]
    
    cdf, cdf_min, cdf_max = lax.fori_loop(0, xplot.size, outer, [xplot, cdf, cdf_min, cdf_max])[1:]
    return cdf, cdf_min, cdf_max



# ================= #
# Utility functions #
# ================= #
@jit
def gamma_unnormalized(n, z):
    return jsc.special.gammaincc(n, z) * jsc.special.gamma(n)

@jit
def _gammaincc_neg_n(n, z):
    """More straightforward implementation, but incompatible with autodiff."""
    # save target value of n and initialize starting point of recurrence relation 
    # with a positive value of n
    n_orig = n
    n %= 1
    
    def cond_f(args):
        return ~jnp.isclose(args[1], n_orig)
        
    # iterate thru recursive relations til target value is reached
    def body_f(args):
        val, n = args
        val = (val - z**(n-1)*jnp.exp(-z)) / (n-1)
        n -= 1
        return [val, n]
    
    val = gamma_unnormalized(n, z)
    val, n = lax.while_loop(cond_f,
                            body_f,
                            [val, n])
    return val

def gammaincc_neg_n(n, z):
    # save target value of n and create iteration array
    n_orig = n
    n = (n%1) - jnp.arange(101)[:,None]

    # iterate thru recursive relations til target value is reached, accounting for precision error
    def body_f(val, n):
        val = lax.cond(jnp.logical_and(~jnp.isclose(n[0], n_orig), (n[0]+.5)>n_orig),
                       lambda val: (val - z**(n-1)*jnp.exp(-z)) / (n-1),
                       lambda val: val,
                       val)
        return val, n

    val = gamma_unnormalized(n[0], z)
    val, n = lax.scan(body_f, val, n)

    # only return first element because autoconversion to sized ndarray occurs
    return val[0]

@jit
def gammaincc(n, z):
    """For scalar inputs.

    Parameters
    ----------
    n : float
    z : float

    Returns
    -------
    float
    """
    y = lax.cond(n>=0,
                 gamma_unnormalized,
                 gammaincc_neg_n,
                 n, z)
    return y

@jit
def expn(n, z):
    n = jnp.squeeze(n)
    z = jnp.squeeze(z)
    return z**(n-1) * gammaincc(1-n, z)



# ======= #
# Classes #
# ======= #
class MeanFieldLKW(pyro.distributions.Distribution):
    arg_constraints = {
        "alpha": constraints.interval(.5, 3.),
        "kappa": constraints.positive,
        "b": constraints.unit_interval,
        "rstar": constraints.positive
    }
    support = pyro.distributions.constraints.positive  # The distribution is defined for positive values
    # reparametrized_params = ["a", "b"]

    def __init__(self, alpha, kappa, b, rstar, r0=jnp.array(1.)):
        """Class for fitting demographic scaling with resource competition correction.
        
        Parameters
        ----------
        alpha : float
            Power law exponent. alpha=1 corresponds to Zipf's law.
        kappa : float
            Resource need exponent.
        b : float
            Metabolic growth exponent.
        rstar : float
            Characteristic radius.
        r0 : float
            Min radius.
        """
        self.alpha = alpha
        self.kappa = kappa
        self.b = b
        self.rstar = rstar
        self.r0 = r0
        self._batch_shape = ()  # No batch dimensions
        self._event_shape = (alpha.size,)

    def model(self, X):
        assert X.min()>=self.r0
        return self.log_likelihood(X)

    def log_prob(self, X):
        return self.log_likelihood(X)

    def cdf(self, X):
        Y = jnp.zeros_like(X)
        
        tot_exp = self.kappa + 1. - self.b
        term1 = self.r0**-self.alpha * expn(1 + self.alpha / tot_exp, (self.r0/self.rstar)**tot_exp / tot_exp)
        for i in range(X.size):            
            term2 = X[i]**-self.alpha * expn(1 + self.alpha / tot_exp, (X[i]/self.rstar)**tot_exp / tot_exp)
            Y = Y.at[i].set((term1 - term2) / term1)
        return Y

    @jit
    def icdf(self, c):
        Y = jnp.zeros_like(c)
        for i, c_ in enumerate(c):
            # transform radius into translated and then logarithmic variable for smooth solution
            cost = lambda logx: (jnp.squeeze(self.cdf(jnp.exp(logx)+self.r0)) - c_)**2
            sol = minimize(cost, jnp.array([0.]), method='BFGS')
            Y = Y.at[i].set(jnp.exp(jnp.squeeze(sol.x))+self.r0)
        return Y
            
    @jit
    def log_likelihood(self, X):
        """Log likelihood of observations in self.X given model parameters.

        Parameters
        ----------
        args : list-like
            Logarithm of model parameters in order of alpha, kappa, b, rstar. 
        
        Returns
        -------
        float
        """
        delta = self.kappa + 1. - self.b
        return ((jnp.log(X) - jnp.log(self.r0)) * -(self.alpha + 1) - 
                (X/self.rstar)**delta / delta -
                jnp.log(self.Z()))

    def pdf(self, X):
        return jnp.exp(self.log_likelihood(X))

    @jit
    def Z(self):
        """Normalization constant.
        
        Parameters
        ----------
        alpha : float
            Demographic exponent.
        kappa : float
            Fluctuations exponent.
        b : float
            Metabolic growth exponent.
        rstar : float
        r0 : float
        """
        tot_exp = self.kappa + 1. - self.b
        Z = self.r0  / tot_exp * expn(1 + self.alpha / tot_exp,
                                      (self.r0/self.rstar)**tot_exp / tot_exp)
        return Z

    def sample(self, key, sample_shape=()):
        """Rejection sampling approach, using power law as proposal distribution."""
        n_sample = 1
        for d in sample_shape:
            n_sample *= d
        alpha = self.alpha
        r0 = self.r0
        X = jnp.zeros(n_sample)
        r0_scale = self.pdf(r0)
    
        def cond_f(val):
            key, counter, X = val
            return counter < n_sample
    
        def update(val):
            counter, r, X = val
            X = X.at[counter].set(r)
            counter += 1
            return [counter, r, X]
        
        def body_f(val):
            key, counter, X = val
            # sample from proposal distribution (a power law)
            key, subkey = random.split(key)
            r = self.sample_pl(subkey, alpha, r0)
            
            # rejection step
            key, subkey = random.split(key)
            # rescale height of proposal distribution
            u = random.uniform(subkey) * r**(-alpha-1) / r0**-alpha * r0 * r0_scale
            counter, r, X = lax.cond(u <= self.pdf(r),
                                     update,
                                     lambda val: val,
                                     [counter, r, X])
            return [key, counter, X]
    
        key, counter, X = lax.while_loop(cond_f,
                                         body_f,
                                         [key, 0, X])
            
        return X.reshape(sample_shape)

    @classmethod
    def sample_pl(cls, key, alpha, xmin, shape=()):
        u = random.uniform(key, shape=shape)
        return xmin * (1 - u) ** (-1 / alpha)
    
    def __str__(self):
        return f'alpha = {self.alpha}\nkappa = {self.kappa}\nb = {self.b}\nrstar = {self.rstar}'

    # PyTree methods
    def tree_flatten(self):
        return (self.alpha, self.kappa, self.b, self.rstar, self.r0), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        alpha, kappa, b, rstar, r0 = children
        return cls(alpha, kappa, b, rstar, r0)
#end MeanFieldLKW

