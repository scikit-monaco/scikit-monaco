
import numpy as np
import numpy.random

from _mc import integrate_importance
from mc_base import _MC_Base

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
        _MC_Base.__init__(self,rng,nprocs,seed,batch_size)

    def get_ndims(self):
        generated_points = self.distribution(size=1,**self.dist_kwargs)
        return np.size(generated_points)

    def make_integrator(self):
        def func(batch_number):
            seed = self.get_seed_for_batch(batch_number)
            return integrate_importance(self.f,self.npoints,
                    self.distribution,self.args,self.rng,seed,
                    self.dist_kwargs,self.weight)
        return func


def mcimport(f,npoints,distribution,args=(),dist_kwargs={},
        rng=numpy.random,nprocs=None,seed=None,batch_size=None,weight=1.0):
    return _MC_Importance_Integrator(f,npoints,distribution,args=args,
            dist_kwargs=dist_kwargs,rng=rng,nprocs=nprocs,
            seed=seed,batch_size=batch_size,weight=weight).run()

