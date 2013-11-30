
from __future__ import division, absolute_import

import numpy as np
import multiprocessing
import skmonaco.mp as mp

class _MC_Base(object):

    default_batch_size = 10000

    def __init__(self,nprocs,batch_size):
        if nprocs is None:
            self.nprocs = multiprocessing.cpu_count()
        else:
            self.nprocs = nprocs
        if batch_size is None:
            self.batch_size = self.default_batch_size
        else:
            self.batch_size = int(batch_size)
            if self.batch_size < 1:
                raise ValueError("'batch_size' must be >= 1.")
        self.batches = self.create_batches()

    @property
    def nbatches(self):
        return len(self.batch_sizes)

    def create_batches(self):
        if self.npoints < self.batch_size:
            # Form a single batch
            self.batch_sizes = [ self.npoints ]
        else:
            # More than one batch
            if self.npoints % self.batch_size < 0.1*self.batch_size:
                # Last batch would be too small: lump it with previous
                # batch
                nbatches = self.npoints // self.batch_size-1
                remainder = self.batch_size + self.npoints % self.batch_size
            else:
                # Last batch is big enough
                nbatches = self.npoints // self.batch_size
                remainder = self.npoints % self.batch_size
            self.batch_sizes = [ self.batch_size ]*nbatches + [remainder]

    def run_serial(self):
        summ, sum2 = 0., 0.
        f = self.make_integrator()
        for ibatch,batch_size in enumerate(self.batch_sizes):
            batch_sum, batch_sum2 = f(ibatch)
            summ += batch_sum
            sum2 += batch_sum2
        return summ/self.npoints, \
                np.sqrt(sum2-summ**2/self.npoints)/self.npoints

    def run_parallel(self):
        f = self.make_integrator()
        res = mp.parmap(f,range(self.nbatches),nprocs=self.nprocs)
        summ, sum2s = zip(*res)
        summ = np.array(summ).sum(axis=0)
        sum2 = np.array(sum2s).sum(axis=0)
        return summ/self.npoints, \
                np.sqrt(sum2-summ**2/self.npoints)/self.npoints

    def run(self):
        if self.nprocs == 1:
            return self.run_serial()
        else:
            return self.run_parallel()

