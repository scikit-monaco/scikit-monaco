
from __future__ import division, absolute_import

import numpy as np
import numpy.random

from . import _mc
from .mc_base import _MC_Base
import skmonaco.random_utils as random_utils

__all__ = [ "mcimport" ]

class _MC_Importance_Integrator(_MC_Base):

    def __init__(self,f,npoints,distribution,
            args, rng, nprocs,seed,batch_size,
            dist_kwargs,weight):
        self.f = f
        self.npoints = int(npoints)
        self.distribution = distribution
        self.dist_kwargs = dist_kwargs
        self.ndims = self.get_ndims()
        self.weight = weight
        self.args = args
        if rng is None:
            self.rng = numpy.random
        else:
            self.rng = rng
        self.seed_generator = random_utils.SeedGenerator(seed)
        _MC_Base.__init__(self,nprocs,batch_size)

    def get_ndims(self):
        generated_points = self.distribution(size=1,**self.dist_kwargs)
        return np.size(generated_points)

    def make_integrator(self):
        def func(batch_number):
            seed = self.seed_generator.get_seed_for_batch(batch_number)
            return _mc.integrate_importance(self.f,self.batch_sizes[batch_number],
                    self.distribution,self.args,self.rng,seed,
                    self.dist_kwargs,self.weight)
        return func


def mcimport(f,npoints,distribution,args=(),dist_kwargs={},
        rng=None,nprocs=1,seed=None,batch_size=None,weight=1.0):
    """
    Compute a definite integral, sampling from a non-uniform distribution.

    This routine integrates ``f(x)*distribution(x)`` by drawing samples from
    `distribution`. Choosing `distribution` such that the variance of `f` is
    small will lead to much faster convergence than just using uniform
    sampling.

    Parameters
    ----------
    f : function
        A Python function or method to integrate. It must take an iterable
        of length `d`, where `d` is the dimensionality of the integral,
        as argument, and return either a float or a numpy array.
    npoints : int >= 2
        Number of points to use in the integration.
    distribution : function
        A Python function or method which returns random points. 
        ``distribution(size) -> numpy array of shape (size,d)`` where
        `d` is the dimensionality of the integral. If ``d==1``, 
        distribution can also return an array of shape ``(size,)``.
        The module `numpy.random` contains a large number of 
        distributions that can be used here.

    Other Parameters
    ----------------
    args : tuple, optional
        Extra arguments to be passed to `f`.
    dist_kwargs : dictionary, optional
        Keyword arguments to be passed to `distribution`.
    nprocs : int >= 1, optional 
        Number of processes to use for the integration. 1 by default.
    seed : int, iterable, optional
        Seed for the random number generator. Running the integration with the
        same seed guarantees that identical results will be obtained (even
        if nprocs is different). 
        If the argument is absent, this lets the random number generator
        handle the seeding.
        If the default rng is used, this means the seed will be read from
        `/dev/random`.
    rng : module or class, optional
        Random number generator. Must expose the attributes `seed` by
        default. The ``numpy.random`` module by default.
    batch_size : int, optional
        The integration is batched, meaning that `batch_size` points 
        are generated, the integration is run with these points, and 
        the results are stored. Each batch is run by a single process. 
        It may be useful to reduce `batch_size` if the 
        dimensionality of the integration is very large.
    weight : float, optional
        Multiply the result and error by this number. 1.0 by default. 
        This can be used when the measure of the integral is not 1.0.
        For instance, if one is sampling from a uniform distribution,
        the integration volume could be passed to `weight`.

    Returns
    -------
    value : float or numpy array
        The estimate for the integral. If the integrand returns an
        array, this will be an array of the same shape.
    error : float or numpy array
        The standard deviation of the result. If the integrand 
        returns an array, this will be an array of the same shape.

    Examples
    --------
    Suppose that we want to integrate ``exp(-x**2/2)`` from x = -1 to 1. 
    We can sample from the normal distribution, such that the
    function ``f`` is ``f = sqrt(2*pi) if -1. < x < 1. else 0``.

    >>> import numpy as np
    >>> from numpy.random import normal
    >>> f = lambda x: np.sqrt(2*np.pi) * (-1. < x < 1.)
    >>> npoints = 1e5
    >>> mcimport(f,npoints,normal)
    (1.7119..., 0.00116...)
    >>> from scipy.special import erf
    >>> np.sqrt(2.*np.pi) * erf(1/np.sqrt(2.)) # exact value
    1.7112...

    Now suppose that we want to integrate exp(z^2) in the unit 
    sphere (x^2 + y^2 + z^2 < 1). Since the integrand is uniform
    along x and y and normal along z, we choose to sample uniformly
    from x and y and normally along z. We can hasten the integration by 
    using symmetry and only considering the octant (x,y,z > 0).

    >>> import numpy as np
    >>> from numpy.random import normal, uniform
    >>> f = lambda (x,y,z): np.sqrt(2.*np.pi)*(z>0.)*(x**2+y**2+z**2<1.)
    >>> def distribution(size):
    ...     xs = uniform(size=size)
    ...     ys = uniform(size=size)
    ...     zs = normal(size=size)
    ...     return np.array((xs,ys,zs)).T
    >>> npoints = 1e5
    >>> result, error = mcimport(f,npoints,distribution)
    >>> result*8,error*8
    (3.8096..., 0.0248...)

    The integrand can also return an array. Suppose that we want to 
    calculate the integrals of both ``exp(z**2)`` and ``z**2*exp(z**2)``
    in the unit sphere. We choose the same distribution as in the 
    previous example, but the function that we sum is now:

    >>> f = lambda (x,y,z): (np.sqrt(2.*np.pi)*(z>0.)*(x**2+y**2+z**2<1.)*
    ...     np.array((1.,z**2)))
    >>> result, error = mcimport(f,npoints,distribution)
    >>> result*8
    array([ 3.81408558,  0.67236413])
    >>> error*8
    array([ 0.02488709,  0.00700179])
    """
    return _MC_Importance_Integrator(f,npoints,distribution,args=args,
            dist_kwargs=dist_kwargs,rng=rng,nprocs=nprocs,
            seed=seed,batch_size=batch_size,weight=weight).run()

