
from __future__ import division, absolute_import

import numpy as np
import numpy.random

import skmonaco.random_utils as random_utils
from .mc_base import _MC_Base
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
        f = self.make_integrator()
        for ibatch,batch_size in enumerate(self.batch_sizes):
            res, std = f(ibatch)
            res_sum += res
            var_sum += std**2
        return res_sum/self.nbatches,np.sqrt(var_sum)/self.nbatches

    def run_parallel(self):
        raise NotImplementedError


def mcmiser(f,npoints,xl,xu,args=(),rng=None,nprocs=1,seed=None,
        min_bisect=100,pre_frac=0.1,exponent=2./3.):
    return _MC_Miser_Integrator(f,npoints,args=args,
            xl=xl,xu=xu,rng=rng,nprocs=nprocs,seed=seed,
            min_bisect=min_bisect,pre_frac=pre_frac,exponent=exponent).run()

