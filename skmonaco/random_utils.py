

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
        If the seed is not specified, this class will return None for all 
        seeds, prompting the rng to handle its own seeding. This should
        be the preferred method of ensuring different runs have different
        seeds.
    """
    def __init__(self,seed=None):
        if seed is None:
            self.dont_seed = True
        else:
            self.dont_seed = False
            try:
                seed + []
            except TypeError:
                if isinstance(seed,float):
                    raise RuntimeWarning(
                        "The numpy RNG automatically converts seeds "
                        "to integers. This may result in a loss of "
                        "precision.")
                seed = [seed]
            self.seed = seed
            self.seed_cache = set()

    def get_seed_for_batch(self,batch_number):
        if self.dont_seed:
            return None
        else:
            seed = tuple(self.seed + [batch_number])
            if seed in self.seed_cache:
                raise RuntimeWarning("The seed generator has produced "
                        "the same seed more than once. Check that it is "
                        "passed unique batch numbers.")
            self.seed_cache.add(seed)
            return seed

