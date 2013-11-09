
import multiprocessing
import numpy as np
import numpy.random
import time
import os

import mp
from _mc import integrate_kernel

__all__ = [ "mcquad" ]

class _MC_Integrator(object):

    default_batch_size = 10000

    def __init__(self,f,npoints,xl,xu,
            rng=numpy.random,nprocs=None,seed=None,batch_size=None):
        self.f = f
        self.npoints = int(npoints)
        self.xl = np.array(xl)
        self.xu = np.array(xu)
        if len(self.xl) != len(self.xu):
            raise ValueError("'xl' and 'xu' must be the same length.")
        if nprocs is None:
            self.nprocs = multiprocessing.cpu_count()
        else:
            self.nprocs = nprocs
        if self.npoints < 2:
            raise ValueError("'npoints' must be >= 2.")
        if batch_size is None:
            self.batch_size = self.default_batch_size
        else:
            self.batch_size = batch_size
            if self.batch_size < 1:
                raise ValueError("'batch_size' must be >= 1.")
        self.batches = self.create_batches()
        self.rng = rng
        if seed is None:
            self.seed = [ time.time(), os.getpid() ]
        else:
            try:
                seed + []
            except TypeError:
                raise TypeError("Seed must be a list.")
            self.seed = seed

    @property
    def nbatches(self):
        return len(self.batches)

    def create_batches(self):
        if self.npoints % self.batch_size < 0.1*self.batch_size:
            nbatches = self.npoints / self.batch_size-1
            remainder = self.batch_size + self.npoints % self.batch_size
        else:
            nbatches = self.npoints / self.batch_size
            remainder = self.npoints % self.batch_size
        return [ self.batch_size ]*nbatches + [remainder]

    def get_seed_for_batch(self,batch_number):
        return self.seed + [(batch_number*2661+36979)%175000]

    def make_integrator(self):
        f = self.f
        batches = self.batches
        xl = self.xl
        xu = self.xu
        seed_gen = self.get_seed_for_batch
        def func(batch_number):
            seed = seed_gen(batch_number)
            return integrate_kernel(f,batches[batch_number],
                    xl,xu,rng=self.rng,seed=seed)
        return func

    def run_serial(self):
        summ, var = 0., 0.
        f = self.make_integrator()
        for ibatch,batch in enumerate(self.batches):
            batch_sum, batch_sd = f(ibatch)
            summ += batch_sum*batch
            var += batch_sd**2*batch**2
        return summ/self.npoints, np.sqrt(var)/self.npoints

    def run_parallel(self):
        f = self.make_integrator()
        res = mp.parmap(f,range(self.nbatches),nprocs=self.nprocs)
        summ, sds = zip(*res)
        summ = np.array(summ)
        sds = np.array(sds)
        batches = np.array(self.batches)
        return sum(summ*batches)/self.npoints, np.sqrt(sum(batches**2*sds**2))/self.npoints

    def run(self):
        if self.nprocs == 1:
            return self.run_serial()
        else:
            return self.run_parallel()

def mcquad(f,npoints,xl,xu,rng=numpy.random,nprocs=None,
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

    Returns
    -------

    value : float
        The estimate for the integral.
    error : float
        An estimate for the error (the integral has a 0.68 probability of
        being within `error` of the correct answer).

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

    Examples
    --------

    Integrate x*y over the unit square. The true value is 1./4.

    >>> mcquad(lambda x: x[0]*x[1], npoints=20000, xl=[0.,0.],xu=[1.,1.])
    (0.24966..., 0.0015488)

    Calculate pi/4 by summing over all points in the unit circle that 
    fall within 1 of the origin.

    >>> mcquad(lambda x: 1 if sum(x**2) < 1 else 0.,
    ...     npoints=20000, xl=[0.,0.], xu=[0.,0.])
    (0.78550..., 0.0029024...)
    >>> np.pi/4.
    0.7853981633974483
    """
    return _MC_Integrator(f,npoints,xl,xu,rng,nprocs,seed,batch_size).run()
    
