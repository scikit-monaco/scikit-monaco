
from numpy.testing import TestCase
import pickle
from numpy.random import randint
from utils import run_module_suite

import miser_fixture


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

if __name__ == '__main__':
    import sys
    run_module_suite(argv=sys.argv)
