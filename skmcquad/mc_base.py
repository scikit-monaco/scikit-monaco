import numpy as np
import multiprocessing
import mp
import time
import os

class _MC_Base(object):

    default_batch_size = 10000

    def __init__(self,rng,nprocs,seed,batch_size):
        self.rng = rng
        if seed is None:
            self.seed = [ time.time(), os.getpid() ]
        else:
            try:
                seed + []
            except TypeError:
                raise TypeError("Seed must be a list.")
            self.seed = seed
        if nprocs is None:
            self.nprocs = multiprocessing.cpu_count()
        else:
            self.nprocs = nprocs
        if batch_size is None:
            self.batch_size = self.default_batch_size
        else:
            self.batch_size = batch_size
            if self.batch_size < 1:
                raise ValueError("'batch_size' must be >= 1.")
        self.batches = self.create_batches()

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

