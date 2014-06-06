
from numpy.testing import TestCase
import pickle
from numpy.random import randint
from utils import run_module_suite

import miser_fixture

from skmonaco import mcmiser


class TestMCMiser(TestCase):

    def setUp(self):
        with open("miser_data.pkl") as f:
            self.fixtures = pickle.load(f)

    def testMiserSavedData(self):
        """
        MISER tests against saved data.
        """
        for fixture in self.fixtures:
            seed = randint(10000)
            fixture.check_trial_run(seed=seed)

    def test_seed(self):
        """
        Test same seed -> same result in MISER
        """
        func = lambda x: x**2
        npoints = 50000
        res, error = mcmiser(func, npoints, [0.], [1.], seed=[1234,5678])
        res2, error2 = mcmiser(func, npoints, [0.], [1.], seed=[1234,5678])
        assert res == res2, "{0} != {1}".format(res, res2)
        assert error == error2
        

if __name__ == '__main__':
    import sys
    run_module_suite(argv=sys.argv)
