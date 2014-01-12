
from __future__ import division, absolute_import

import numpy as np
import numpy.random

from . import _mc
from .mc_base import _MC_Base
import skmonaco.random_utils as random_utils

__all__ = [ "mcquad" ]

class _MC_Integrator(_MC_Base):

    def __init__(self,f,npoints,xl,xu,args=(),
            rng=None,nprocs=1,seed=None,batch_size=None):
        self.f = f
        self.npoints = int(npoints)
        self.xl = np.array(xl)
        self.xu = np.array(xu)
        if rng is None:
            self.rng = numpy.random
        else:
            self.rng = rng
        self.args = args
        if len(self.xl) != len(self.xu):
            raise ValueError("'xl' and 'xu' must be the same length.")
        if self.npoints < 2:
            raise ValueError("'npoints' must be >= 2.")
        self.seed_generator = random_utils.SeedGenerator(seed)
        _MC_Base.__init__(self,nprocs,batch_size)

    def make_integrator(self):
        f = self.f
        batches = self.batch_sizes
        xl = self.xl
        xu = self.xu
        def func(batch_number):
            seed = self.seed_generator.get_seed_for_batch(batch_number)
            return _mc.integrate_uniform(f,batches[batch_number],
                    xl,xu,args=self.args,rng=self.rng,seed=seed)
        return func


def mcquad(f,npoints,xl,xu,args=(),rng=None,nprocs=1,
        seed=None,batch_size=None):
    """
    Compute a definite integral.

    Integrate `f` in a hypercube using a uniform Monte-Carlo method.
    
    Parameters
    ----------
    f : function
        A Python function or method to integrate. It must take an iterable
        of length `d`, where `d` is the dimensionality of the integral,
        as argument, and return either a float or a numpy array.
    npoints : int
        Number of points to use in integration.
    xl, xu : iterable
        Iterable of length d denoting the bottom left corner and upper 
        right corner of the integration region.

    Other Parameters
    ----------------
    nprocs : int >= 1, optional
        Number of processes to use concurrently for the integration. Use 
        nprocs=1 to force a serial evaluation of the integral. This defaults
        to 1.
    seed : int, iterable or None
        Seed for the random number generator. Running the integration with the
        same seed guarantees that identical results will be obtained (even
        if nprocs is different). 
        If the argument is absent, this lets the random number generator
        handle the seeding.
        If the default rng is used, this means the seed will be read from
        `/dev/random`.
    rng : module or class, optional
        Random number generator. Must expose the attributes `seed` and `ranf`.
        The ``numpy.random`` module by default.
    batch_size : int, optional
        The integration is batched, meaning that `batch_size` points are 
        generated, the integration is run with these points, and the 
        results are stored. Each batch is run by a single process. It may 
        be useful to reduce `batch_size` if the dimensionality of the 
        integration is very large.

    Returns
    -------
    value : float or numpy array.
        The estimate for the integral. If the integrand returns an array,
        this will be an array of the same shape.
    error : float or numpy array
        An estimate for the error (the integral has, approximately, a 0.68 
        probability of being within `error` of the correct answer).


    Examples
    --------
    Integrate x*y over the unit square. The true value is 1./4.

    >>> mcquad(lambda x: x[0]*x[1], npoints=20000, xl=[0.,0.],xu=[1.,1.])
    (0.24966..., 0.0015488...)

    Calculate pi/4 by summing over all points in the unit circle that 
    fall within 1 of the origin.

    >>> mcquad(lambda x: 1 if sum(x**2) < 1 else 0.,
    ...     npoints=20000, xl=[0.,0.], xu=[1.,1.])
    (0.78550..., 0.0029024...)
    >>> np.pi/4.
    0.7853981633974483

    The integrand can return an array. This can be used to calculate 
    several integrals at once.

    >>> result, error = mcquad(
    ...      lambda x: np.exp(-x**2)*np.array((1.,x**2,x**4,x**6)),
    ...     npoints=20000, xl=[0.], xu=[1.])
    >>> result
    array([ 0.7464783 ,  0.18945015,  0.10075603,  0.06731908])
    >>> error
    array([ 0.0014275 ,  0.00092622,  0.00080145,  0.00069424])
    """
    return _MC_Integrator(f,npoints,args=args,
            xl=xl,xu=xu,rng=rng,nprocs=nprocs,seed=seed,batch_size=batch_size).run()
    
