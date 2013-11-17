
import numpy as np
import numpy.random

from _mc import integrate_uniform
from mc_base import _MC_Base
import random_utils

__all__ = [ "mcquad" ]

class _MC_Integrator(_MC_Base):

    def __init__(self,f,npoints,xl,xu,args=(),
            rng=numpy.random,nprocs=None,seed=None,batch_size=None):
        self.f = f
        self.npoints = int(npoints)
        self.xl = np.array(xl)
        self.xu = np.array(xu)
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
            return integrate_uniform(f,batches[batch_number],
                    xl,xu,args=self.args,rng=self.rng,seed=seed)
        return func


def mcquad(f,npoints,xl,xu,args=(),rng=numpy.random,nprocs=None,
        seed=None,batch_size=None):
    """
    Compute a definite integral.

    Integrate `f` in a hypercube using a uniform Monte-Carlo method.
    
    Parameters
    ----------
    f : function
        A Python function or method to integrate. It must take an iterable
        of length `d`, where `d` is the dimensionality of the integral,
        as argument, and return a single value.
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
        to the value returned by multiprocessing.cpu_count().
    seed : iterable, optional
        Seed for the random number generator. Running the integration with the
        same seed guarantees that identical results will be obtained (even
        if nprocs is different). Chooses a value based on the system time
        and process id by default.
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
    value : float
        The estimate for the integral.
    error : float
        An estimate for the error (the integral has a 0.68 probability of
        being within `error` of the correct answer).


    Examples
    --------

    Integrate x*y over the unit square. The true value is 1./4.

    >>> mcquad(lambda x: x[0]*x[1], npoints=20000, xl=[0.,0.],xu=[1.,1.])
    (0.24966..., 0.0015488...)

    Calculate pi/4 by summing over all points in the unit circle that 
    fall within 1 of the origin.

    >>> mcquad(lambda x: 1 if sum(x**2) < 1 else 0.,
    ...     npoints=20000, xl=[0.,0.], xu=[0.,0.])
    (0.78550..., 0.0029024...)
    >>> np.pi/4.
    0.7853981633974483
    """
    return _MC_Integrator(f,npoints,args=args,
            xl=xl,xu=xu,rng=rng,nprocs=nprocs,seed=seed,batch_size=batch_size).run()
    
