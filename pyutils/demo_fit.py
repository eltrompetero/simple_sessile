# ====================================================================================== #
# Module for fitting demographic data curves.
# 
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.stats import lognorm
from mpmath import expint
import numpy as np


class MeanFieldFitter():
    def __init__(self, X, r0=1.):
        """Class for fitting demographic scaling with resource competition correction.
        
        Parameters
        ----------
        X : ndarray
            Observations.
        r0 : float
            Min radius.
        """
        assert X.min()>=r0
        
        self.X = X
        self.r0 = r0
        
    def log_likelihood(self, args):
        """Log likelihood of observations in self.X given model parameters.

        Parameters
        ----------
        args : list-like
            Logarithm of model parameters in order of alpha-1, kappa, b, F. 
        
        Returns
        -------
        float
        """
        alpha, kappa, b, F = np.exp(args)
        
        if (kappa + 1. - b)<= 0: return -np.inf
        
        return ((np.log(self.X) - np.log(self.r0)) * -alpha - 
                F / (kappa + 1. - b) * self.X**(kappa + 1. - b) -
                np.log(self.Z(alpha, kappa, b, F, self.r0))).sum()
        
    def fit(self, alpha=None,
            initial_guess=[2, 1, 1/3, .1],
            full_output=False,
            log_prior=None):
        """Fit via max likelihood.
        
        Parameters
        ----------
        alpha : float
            Demographic exponent.
        initial_guess : list-like, [2, 1, 1/3, .1]
        full_output : bool, False
        log_prior : None, list
            Prior functions for alpha, kappa, b, F.
        
        Returns
        -------
        float
            Best fit demographic exponent alpha.
        float
            Best fit fluctuations exponent kappa.
        float
            Best fit metabolic growth exponent b.
        float
            Best fit resource competition exponent F.
        """
        assert len(initial_guess)==4
        initial_guess = np.log(initial_guess)
        bounds = None
        
        if alpha is None:
            soln = minimize(lambda args: -self.log_likelihood(args),
                            initial_guess,
                            bounds=bounds,
                            method='powell',
                            tol=1e-7)
            alpha, kappa, b, F = np.exp(soln['x'])
            if full_output:
                return (alpha, kappa, b, F), soln
            return alpha, kappa, b, F
        
        fitfun = lambda args: -self.log_likelihood(np.concatenate(([np.log(alpha)],args[1:])))
        soln = minimize(fitfun, initial_guess)
        kappa, b, F = np.exp(soln['x'])[1:]
        if full_output:
            return (kappa, b, F), soln
        return kappa, b, F
   
    def fit_fixb(self, alpha=None,
                 initial_guess=[2, 1, .1],
                 b=1/3,
                 full_output=False,
                 log_prior=None):
        """Fit via max likelihood with b=1/3."""
        assert len(initial_guess)==3
        initial_guess = np.log([initial_guess[0]-1, initial_guess[1], initial_guess[2]])
        bounds = None
        
        if alpha is None:
            if log_prior is None:
                soln = minimize(lambda args: -self.log_likelihood([args[0],args[1],np.log(b),args[2]]),
                                initial_guess,
                                bounds=bounds,
                                method='powell',
                                tol=1e-7)
            else:
                soln = minimize(lambda args: -self.log_likelihood([args[0],args[1],np.log(b),args[2]]) - 
                                log_prior[0](np.exp(args[0])) - 
                                log_prior[1](np.exp(args[1])) - 
                                log_prior[2](np.exp(args[2])),
                                initial_guess,
                                bounds=bounds,
                                method='powell',
                                tol=1e-7)
            alpha, kappa, F = np.exp(soln['x'])
            if full_output:
                return (alpha, kappa, F), soln
            return alpha, kappa, F

        fitfun = lambda args: -self.log_likelihood([np.log(alpha-1),args[1],np.log(b),args[2]])
        soln = minimize(fitfun, initial_guess)
        kappa, F = np.exp(soln['x'])[1:]
        if full_output:
            return (kappa, F), soln
        return kappa, F
        
    @classmethod
    def Z(cls, alpha, kappa, b, F, r0):
        """Normalization constant.
        
        Parameters
        ----------
        alpha : float
            Demographic exponent.
        kappa : float
            Fluctuations exponent.
        b : float
            Metabolic growth exponent.
        F : float
        r0 : float
        """
        tot_exp = kappa + 1. - b
        try:
            Z = r0  / tot_exp * expint(1 + (alpha - 1.) / tot_exp,
                                       F / tot_exp * r0**tot_exp)
            Z = float(Z)
        except TypeError:
            Z = float(Z.real)
        return Z
    
    @classmethod
    def pdf(cls, r, alpha, kappa, b, F, r0=1.):
        """Probability density function."""
        if hasattr(r, '__len__'):
            assert r.min()>=r0
        else:
            assert r>=r0
        
        tot_exp = kappa + 1. - b
        assert tot_exp>0
        
        Z = cls.Z(alpha, kappa, b, F, r0)
        
        return (r / r0)**-alpha * np.exp(-F / tot_exp * r**tot_exp) / Z

    def standard_fit(self):
        """Implement fit with standard lognormal posteriors for model parameters
        and fixed b=1/3.

        Returns
        -------
        float
            Best fit demographic exponent alpha.
        float
            Best fit fluctuations exponent kappa.
        float
            Best fit resource competition exponent F.
        """
        m = 2  # linear mean
        ls = 1  # log std
        lm = np.log(m) - ls**2/2  # log mean
        s = np.sqrt((np.exp(ls**2)-1) * np.exp(2*lm+ls**2))  # linear std
        alpha_prior = lambda alpha, ls=ls, lm=lm: np.log(lognorm.pdf(alpha, s=ls, scale=np.exp(lm)))

        m = .5  # linear mean
        ls = 1  # log std
        lm = np.log(m) - ls**2/2  # log mean
        s = np.sqrt((np.exp(ls**2)-1) * np.exp(2*lm+ls**2))  # linear std
        kappa_prior = lambda alpha, ls=ls, lm=lm: np.log(lognorm.pdf(alpha, s=ls, scale=np.exp(lm)))

        m = 1e-4  # linear mean
        ls = 1  # log std
        lm = np.log(m) - ls**2/2  # log mean
        s = np.sqrt((np.exp(ls**2)-1) * np.exp(2*lm+ls**2))  # linear std
        F_prior = lambda alpha, ls=ls, lm=lm: np.log(lognorm.pdf(alpha, s=ls, scale=np.exp(lm)))

        log_priors = [alpha_prior, kappa_prior, F_prior]

        return self.fit_fixb(initial_guess=[2., 1., .01], log_prior=log_priors)
#end MeanFieldFitter


class WeibullFitter():
    def __init__(self, X):
        """Class for fitting demographic scaling with resource competition correction.
        
        Parameters
        ----------
        X : ndarray
            Observations.
        """
        
        assert X.min()>=0
        
        self.X = X
        
    def log_likelihood(self, args):
        """Log likelihood of observations in self.X given model parameters.
        
        Returns
        -------
        float
        """
        
        el, k = np.exp(args)
        return (np.log(k/el) + (k-1) * np.log(self.X/el) - (self.X/el)**k).sum()

    def fit(self,
            initial_guess=[1., 1.],
            full_output=False):
        """Fit via max likelihood."""
        
        assert len(initial_guess)==2
        initial_guess = np.log(initial_guess)
        bounds = None
        
        soln = minimize(lambda args: -self.log_likelihood(args),
                        initial_guess,
                        bounds=bounds,
                        method='powell')
        el, k = np.exp(soln['x'])
        if full_output:
            return (el, k), soln
        return el, k

    def line_fit(self, xfit, yfit,
                 initial_guess=[1., 1.],
                 full_output=False,
                 log_cost=True):
        """Fit by least squares.

        Parameters
        ----------
        xfit : ndarray
            Dependent variable (radius).
        yfit : ndarray
            Probability assigned to each xfit.
        initial_guess : ndarray, [1., 1.]
        full_output : bool, False
        log_cost : bool, True

        Returns
        -------
        float
        float
        """
        
        assert len(initial_guess)==2
        initial_guess = np.log(initial_guess)
        bounds = None
        
        if log_cost:
            soln = minimize(lambda args: np.linalg.norm(np.log(yfit) -
                                                self.pdf(xfit, np.exp(args[0]), np.exp(args[1]), log=True)),
                            initial_guess,
                            bounds=bounds)
        else:
            soln = minimize(lambda args: np.linalg.norm(yfit -
                                                self.pdf(xfit, np.exp(args[0]), np.exp(args[1]), log=False)),
                            initial_guess,
                            bounds=bounds)
        el, k = np.exp(soln['x'])
        if full_output:
            return (el, k), soln
        return el, k

    @classmethod
    def pdf(cls, r, el, k, log=False):
        """Probability density function."""
        
        assert el>0 and k>=0

        if hasattr(r, '__len__'):
            assert r.min()>=0
        else:
            assert r>=0
        
        logp = np.log(k/el) + (k-1) * np.log(r/el) - (r/el)**k
        if log: return logp
        return np.exp(logp)
#end WeibullFitter


class MeanFieldModel():
    def __init__(self, alpha, kappa, b, F, r0=1):
        assert (kappa + 1. - b)>0

        self.alpha = alpha
        self.kappa = kappa
        self.b = b
        self.F = F
        self.r0 = r0

    def log_likelihood(self, X):
        """Log likelihood of observations in X.
        
        Parameters
        ----------
        X : float or ndarray

        Returns
        -------
        float or ndarray
        """
        if hasattr(X, '__len__'):
            assert X.ndim==1 and (X>=self.r0).all()
            logp = np.zeros(X.size)
            logp = ((np.log(X) - np.log(self.r0)) * -self.alpha - 
                    self.F / (self.kappa + 1. - self.b) * X**(self.kappa + 1. - self.b) -
                    np.log(self.Z()))
            return logp
    
        assert X>=self.r0
        return ((np.log(X) - np.log(self.r0)) * -self.alpha - 
                self.F / (self.kappa + 1. - self.b) * X**(self.kappa + 1. - self.b) -
                np.log(self.Z())).sum()

    def Z(self):
        """Normalization constant.
        """
        tot_exp = self.kappa + 1. - self.b
        try:
            Z = self.r0  / tot_exp * expint(1 + (self.alpha - 1.) / tot_exp,
                                       self.F / tot_exp * self.r0**tot_exp)
            Z = float(Z)
        except TypeError:
            Z = float(Z.real)
        return Z
#end MeanFieldModel 
