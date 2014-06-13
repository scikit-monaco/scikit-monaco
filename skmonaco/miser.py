
from __future__ import division, absolute_import

import numpy as np
import numpy.random

import skmonaco.random_utils as random_utils
from .mc_base import _MC_Base
import skmonaco.mp as mp
from . import _miser

__all__ = ["mcmiser"]

class _MC_Miser_Integrator(_MC_Base):

    def __init__(self,f,npoints,xl,xu,args,rng,nprocs,seed,
            min_bisect,pre_frac,exponent):
        self.f = f
        self.npoints = int(npoints)
        self.xl = np.array(xl)
        self.xu = np.array(xu)
        self.nprocs = nprocs
        self.args = args
        self.min_bisect = min_bisect
        self.pre_frac = pre_frac
        self.exponent = exponent
        if rng is None:
            self.rng = numpy.random
        else:
            self.rng = rng
        if len(self.xl) != len(self.xu):
            raise ValueError("'xl' and 'xu' must be the same length.")
        if self.npoints < 2:
            raise ValueError("'npoints' must be >= 2.")
        self.seed_generator = random_utils.SeedGenerator(seed)
        _MC_Base.__init__(self,nprocs,self.npoints//nprocs)

    def make_integrator(self):
        f = self.f
        batches = self.batch_sizes
        xl = self.xl
        xu = self.xu
        def func(batch_number):
            seed = self.seed_generator.get_seed_for_batch(batch_number)
            batch_size = batches[batch_number]
            res, std = _miser.integrate_miser(f,batch_size,
                    xl,xu,args=self.args,min_bisect=self.min_bisect,
                    pre_frac=self.pre_frac,exponent=self.exponent,
                    rng=self.rng,seed=seed)
            return res,std
        return func

    def run_serial(self):
        res_sum, var_sum = 0., 0.
        assert len(set(self.batch_sizes))==1
        assert len(self.batch_sizes) == 1
        f = self.make_integrator()
        for ibatch,batch_size in enumerate(self.batch_sizes):
            res, std = f(ibatch)
            res_sum += res
            var_sum += std**2
        return res_sum/self.nbatches,np.sqrt(var_sum)/self.nbatches

    def run_parallel(self):
        f = self.make_integrator()
        res_sum, var_sum = 0., 0.
        assert len(set(self.batch_sizes))==1
        assert len(self.batch_sizes) == self.nprocs
        res_list = mp.parmap(f,range(self.nbatches),nprocs=self.nprocs)
        for  res, std in res_list:
            res_sum += res
            var_sum += std**2
        return res_sum/self.nbatches,np.sqrt(var_sum)/self.nbatches


def mcmiser(f,npoints,xl,xu,args=(),rng=None,nprocs=1,seed=None,
        min_bisect=100,pre_frac=0.1,exponent=2./3.):
    """
    Compute a definite integral.

    Integrate `f` in a hypercube using the MISER algorithm.

    Parameters
    ----------
    f : function
        The integrand. Must take an iterable of length `d`, where
        `d` is the dimennsionality of the integral, as argument,
        and return a float.
    npoints : int
        Number of points to use for the integration.
    xl, xu : iterable
        Iterable of length `d`, where `d` is the dimensionality of the 
        integrand, denoting the bottom left corner and upper right 
        corner of the integration region.

    Other Parameters
    ----------------
    nprocs : int >= 1, optional
        Number of processes to use concurrently for the integration. Use 
        nprocs=1 to force a serial evaluation of the integral. This defaults
        to 1. Increasing `nprocs` will increase the stochastic error
        for a fixed number of samples (the algorithm just runs several
        MISER runs in parallel).
    seed : int, iterable or None
        Seed for the random number generator. Running the integration with the
        same seed and the same `nprocs` guarantees that identical results will be obtained. 
        If the argument is absent, this lets the random number generator
        handle the seeding.
        If the default rng is used, this means the seed will be read from
        `/dev/random`.
    rng : module or class, optional
        Random number generator. Must expose the attributes `seed` and `ranf`.
        The ``numpy.random`` module by default.
    args : list
        List of arguments to pass to `f` when integrating.
    min_bisect : int
        Minimum number of points inn which to run a bisection. If the 
        integrator has a budget of points < `min_bisect` for a region,
        it will fall back onto uniform sampling.
    pre_frac : float
        Fraction of points to use for pre-sampling. The MISER algorithm
        will use this fraction of its budget for a given area to 
        decide how to bisect and how to apportion point budgets.
    exponent : float
        When allocating points to the sub-region, the algorithm will
        give a fraction of points proportional to ``range**exponent``,
        where ``range`` is the range of the integrand in the 
        sub-region (as estimated by using a fraction `pre_frac` of 
        points). Numerical Recipes [NR]_ recommends a fraction of 2/3.

    Returns
    -------
    value : float
        The estimate for the integral. 
    error : float
        An estimate for the error (the integral has, approximately, a 0.68 
        probability of being within `error` of the correct answer).

    Notes
    -----
    Unlike mcquad, the integrand cannot return an array. It must return
    a float. 

    The implementation is that proposed in Numerical Recipes [NR]_: when apportioning 
    points, we use ``|max-min|`` as an estimate of the variance in each sub-area, 
    as opposed to calculating the variance explicitly.

    References
    ----------
    .. [NR] W. H. Press, S. A. Teutolsky, W. T. Vetterling, B. P. Flannery, 
           "Numerical recipes: the art of scientific computing", 3rd edition. 
           Cambridge University Press (2007)

    Examples
    --------
    Integrate x*y over the unit square. The correct value is 1/4.

    >>> mcmiser(lambda x: x[0]*x[1], npoints=20000, xl=[0.,0.], xu=[1.,1.])
    (0.249747..., 0.000170...)

    Note that this is about 10 times more accurate than the equivalent 
    call to `mcquad`, for the same number of points.
    """
    return _MC_Miser_Integrator(f,npoints,args=args,
            xl=xl,xu=xu,rng=rng,nprocs=nprocs,seed=seed,
            min_bisect=min_bisect,pre_frac=pre_frac,exponent=exponent).run()

