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
import scipy.special as scs
from scipy.optimize import minimize
import mpmath as mp
from functools import cache

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
    model : callable
        Model to sample from.
    X : ndarray
        Data to sample from.
    n_loop : int
        Number of iterations to run.
    key : jax.random.PRNGKey, optional
        Random key for sampling, by default None.
    iprint : bool, optional
        Whether to print progress, by default True.
    mcmc_kwargs : dict, optional
        Additional arguments for MCMC, by default {}.
        - num_warmup : int
            Number of warmup steps.
        - num_samples : int
            Number of samples to draw.
        - num_chains : int
            Number of chains to run in parallel.
        - thinning : int
            Thinning interval for samples.
    """
    sample = []
    potential_energy = []
    key = key if not key is None else random.PRNGKey(np.random.randint(2**32-1))

    for i in range(n_loop):
        if iprint: t0 = time.time()
        nuts_kernel = NUTS(model, dense_mass=True, max_tree_depth=6, init_strategy=init_to_sample)
        sampler = MCMC(nuts_kernel, **mcmc_kwargs)
        
        key, subkey = random.split(key)
        sampler.run(subkey, X, extra_fields=('potential_energy',))
        sample.append(sampler.get_samples(True).copy())
        potential_energy.append(sampler.get_extra_fields()['potential_energy'])
        if iprint: print(f"Done with iteration {i+1}/{n_loop} in {time.time()-t0:.2f} seconds.")
    return sample, potential_energy

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

def cdf_bounds_lkw(xplot, params, percentile=(5,95)):
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
        mft = MeanFieldLKW(params['alpha'].ravel()[j],
                           params['kappa'].ravel()[j],
                           params['b'].ravel()[j],
                           params['rstar'].ravel()[j],
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

def cdf_bounds_det(xplot, params, percentile=(5,95)):
    """Return CDF with confidence bounds given the parameter samples along
    the given x-coordinates. This is a helper function for plotting.
    
    Parameters
    ----------
    xplot : jnp.array
    params : dict
        Key 'mu' that gives a list-like object.

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
        mft = DET(params['mu'].ravel()[j],
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

def cdf_bounds_det2(xplot, params, percentile=(5,95)):
    """Return CDF with confidence bounds given the parameter samples along
    the given x-coordinates. This is a helper function for plotting.
    
    Parameters
    ----------
    xplot : jnp.array
    params : dict
        Key 'mu' that gives a list-like object.

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
        mft = DET2(params['mu'].ravel()[j],
                   params['phi'].ravel()[j],
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
def gammaincc_unnormalized(n, z):
    return lax.cond(n==0,
                    lambda _: -0.5772156649015328606 - jnp.log(z) + z - z**2/4 + z**3/18 - z**4/96 + z**5/600,
                    lambda _: jsc.special.gammaincc(n, z) * jsc.special.gamma(n),
                    n)

def gammainc_unnormalized(n, z):
    return jsc.special.gammainc(n, z) * jsc.special.gamma(n)

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
    
    val = gammaincc_unnormalized(n, z)
    val, n = lax.while_loop(cond_f,
                            body_f,
                            [val, n])
    return val

def gammaincc_neg_n(n, z):
    # save target value of n and create iteration array
    n_orig = n
    n = (n%1) - jnp.arange(101)

    # iterate thru recursive relations til target value is reached, accounting for precision error
    def body_f(val, n):
        val = lax.cond(jnp.logical_and(~jnp.isclose(n, n_orig), (n+.5)>n_orig),
                       lambda val: (val - z**(n-1)*jnp.exp(-z)) / (n-1),
                       lambda val: val,
                       val)
        return val, n

    val = gammaincc_unnormalized(n[0], z)
    val, n = lax.scan(body_f, val, n)

    # only return first element because autoconversion to sized ndarray occurs
    return val

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
                 gammaincc_unnormalized,
                 gammaincc_neg_n,
                 n, z)
    return y

@jit
def expn(n, z):
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
    support = constraints.positive  # The distribution is defined for positive values
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
        X : jnp.ndarray
        
        Returns
        -------
        jnp.ndarray
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



class DET(pyro.distributions.Distribution):
    arg_constraints = {
        "mu": constraints.positive
    }
    support = constraints.positive  # The distribution is defined for positive values

    def __init__(self, mu, r0=jnp.array(1.)):
        """Class for fitting demographic scaling with resource competition correction.
        
        Parameters
        ----------
        mu : float
        r0 : float
            Min radius.
        """
        self.mu = mu
        self.r0 = r0
        self._batch_shape = ()  # No batch dimensions
        self._event_shape = (mu.size,)

    def model(self, X):
        assert X.min()>=self.r0
        return self.log_likelihood(X)

    def log_prob(self, X):
        return self.log_likelihood(X)

    def cdf(self, X):
        return 1. - jnp.exp(4*self.mu * (1-(X/self.r0)**(2./3)))

    def icdf(self, c):
        return self.r0 / 8 * ((4*self.mu - jnp.log(1-c)) / self.mu)**(3/2)

    def log_likelihood(self, X):
        """Log likelihood of observations in self.X given model parameters.

        Parameters
        ----------
        X : jnp.ndarray
        
        Returns
        -------
        jnp.ndarray
            Same shape as X.
        """
        return jnp.log(8/3*self.mu/self.r0) - jnp.log(X/self.r0)/3 + 4*self.mu * (1 - (X/self.r0)**(2/3))

    def pdf(self, X):
        return jnp.exp(self.log_likelihood(X))

    def sample(self, key, sample_shape=()):
        u = random.uniform(key, shape=sample_shape)
        return self.icdf(u)
 
    def max_likelihood_mu(self, X):
        """Max likelihood estimate of mu."""
        return 1 / ( 4 * (-1 + jnp.mean((X/self.r0)**(2/3))) )

    def __str__(self):
        return f'mu = {self.mu}'

    # PyTree methods
    def tree_flatten(self):
        return (self.mu, self.r0), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        mu, r0 = children
        return cls(mu, r0)
#end DET


class DET2(pyro.distributions.Distribution):
    arg_constraints = {
        "mu": constraints.positive,
        "phi": constraints.interval(0, 1),
    }
    support = constraints.positive  # The distribution is defined for positive values

    def __init__(self, mu, phi, r0=jnp.array(1.)):
        """Class for fitting demographic scaling with resource competition correction.
        
        Parameters
        ----------
        mu : float
        phi : float
        r0 : float
            Min radius.
        """
        self.mu = mu
        self.phi = phi
        self.r0 = r0
        self._batch_shape = ()  # No batch dimensions
        self._event_shape = (mu.size,)

    def model(self, X):
        assert X.min()>=self.r0
        return self.log_likelihood(X)

    def log_prob(self, X):
        return self.log_likelihood(X)

    def cdf(self, X):
        mu = self.mu
        phi = self.phi
        return 1. - jnp.exp(mu/(1-phi) * (self.r0**(1-phi) - X**(1-phi)))

    def icdf(self, c):
        mu = self.mu
        phi = self.phi
        return (mu/(self.r0**(1 - phi) * mu - jnp.log(1 - c) + phi * jnp.log(1 - c)))**(1/(- 1 + phi))

    def log_likelihood(self, X):
        """Log likelihood of observations in self.X given model parameters.

        Parameters
        ----------
        X : jnp.ndarray
        
        Returns
        -------
        jnp.ndarray
            Same shape as X.
        """
        mu = self.mu
        phi = self.phi
        return jnp.log(mu) - phi*jnp.log(X) + mu/(1-phi)*(self.r0**(1-phi) - X**(1-phi))

    def pdf(self, X):
        return jnp.exp(self.log_likelihood(X))

    def sample(self, key, sample_shape=()):
        u = random.uniform(key, shape=sample_shape)
        return self.icdf(u)
 
    def __str__(self):
        return f'mu = {self.mu}\nphi = {self.phi}'

    # PyTree methods
    def tree_flatten(self):
        return (self.mu, self.phi, self.r0), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        mu, phi, r0 = children
        return cls(mu, phi, r0)
#end DET2


class ExpTruncatedPowerLaw(pyro.distributions.Distribution):
    arg_constraints = {
        "alpha": constraints.interval(0., 3.),
        "el": constraints.positive,
    }
    support = constraints.positive  # The distribution is defined for positive values

    def __init__(self, alpha, el, x0=jnp.array(1.)):
        """Exponentially truncated power law distribution."""
        self.alpha = alpha
        self.el = el
        self.x0 = x0
        self._batch_shape = ()  # No batch dimensions
        self._event_shape = (alpha.size,)

    def model(self, X):
        assert X.min()>=self.x0
        return self.log_likelihood(X)

    def log_prob(self, x):
        return (-(self.alpha+1) * jnp.log(x) - self.el*x - self.alpha*jnp.log(self.el) -
                jnp.log(gammaincc(-self.alpha, self.x0*self.el)))

    def cdf(self, x):
        return 1 - gammaincc(-self.alpha, x*self.el) / gammaincc(-self.alpha, self.x0*self.el)

    def pdf(self, X):
        return jnp.exp(self.log_prob(X))

    def sample(self, key, sample_shape=()):
        """Rejection sampling approach, using power law as proposal distribution."""
        n_sample = 1
        for d in sample_shape:
            n_sample *= d
        alpha = self.alpha
        x0 = self.x0
        X = jnp.zeros(n_sample)
        x0_scale = self.pdf(x0)
    
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
            r = self.sample_pl(subkey, alpha, x0)
            
            # rejection step
            key, subkey = random.split(key)
            # rescale height of proposal distribution
            u = random.uniform(subkey) * r**(-alpha-1) / x0**-alpha * x0 * x0_scale
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
        return f'alpha = {self.alpha}\nel = {self.el}'

    # PyTree methods
    def tree_flatten(self):
        return (self.alpha, self.el, self.x0), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        alpha, el, x0 = children
        return cls(alpha, el, x0)
#end ExpTruncatedPowerLaw
