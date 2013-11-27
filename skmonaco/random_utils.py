
import numpy.random

import time
import os

class SeedGenerator(object):
    """
    Handles the production of unique seeds.

    When running a Monte Carlo calculation in batches,
    each batch must be assigned a unique seed. This
    class takes batch numbers and returns a valid seed
    that is hopefully decorrelated from previous seeds.

    Parameters
    ----------
    seed: int or iterable, optional
        Either a single integer or a list of integers. This seed is
        used to initiate the seed-generation process.
        If not specified, a combination of the system time and
        process ID is used.
    """
    def __init__(self,seed=None):
        if seed is None:
            self.seed = [ time.time(), os.getpid() ]
        else:
            try:
                seed + []
            except TypeError:
                seed = [seed]
            self.seed = seed
        self.seed_cache = set()

    def get_seed_for_batch(self,batch_number):
        seed = tuple(self.seed + [(batch_number*2661+36979)%175000])
        if seed in self.seed_cache:
            raise RuntimeWarning("The seed generator has produced "
                    "the same seed more than once. Check that it is "
                    "passed unique batch numbers.")
        return seed

