
from numpy.testing import TestCase, assert_almost_equal
from utils import assert_within_tol, run_module_suite

import skmonaco.random_utils as random_utils

class TestSeedGenerator(TestCase):

    def test_noseed(self):
        s = random_utils.SeedGenerator()
        assert s.get_seed_for_batch(0) is None

    def test_floatseed(self):
        with self.assertRaises(RuntimeWarning):
            s = random_utils.SeedGenerator(123.4)

    def test_seeded(self):
        s = random_utils.SeedGenerator(123)
        assert s.get_seed_for_batch(6) == (123,6)

    def test_clashing_seeds(self):
        s = random_utils.SeedGenerator(123)
        s.get_seed_for_batch(6)
        with self.assertRaises(RuntimeWarning):
            s.get_seed_for_batch(6)

if __name__ == '__main__':
    import sys
    run_module_suite(argv=sys.argv)
