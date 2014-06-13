
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

    def test_seed_different(self):
        """
        Test different seed -> different result in MISER
        """
        func = lambda x: x**2
        npoints = 50000
        res, error = mcmiser(func, npoints, [0.], [1.], seed=[1234,5678])
        res2, error2 = mcmiser(func, npoints, [0.], [1.], seed=[1235,5678])
        assert res != res2, "{0} == {1}".format(res, res2)
        assert error != error2
        
    def test_wrong_xl(self):
        """
        Raise a ValueError if len(xl) != len(xu) in MISER.
        """
        with self.assertRaises(ValueError):
            mcmiser(lambda x: x**2,2000,xl=[0.,0.],xu=[1.])

    def test_wrong_npoints(self):
        """
        Raise a ValueError if npoints < 2 in MISER.
        """
        with self.assertRaises(ValueError):
            mcmiser(lambda x: x**2,0,xl=[0.],xu=[1.])

    def test_wrong_nprocs(self):
        """
        Raise a ValueError if nprocs < 1
        """
        with self.assertRaises(ValueError):
            mcmiser(lambda x: x**2,2000,xl=[0.],xu=[1.],nprocs=-1)

    def test_integer_seed(self):
        """
        Check that MISER works with an integer seed.
        """
        func = lambda x: x**2
        npoints = 50000
        res, error = mcmiser(func, npoints, [0.], [1.], seed=12345)
        res2, error2 = mcmiser(func, npoints, [0.], [1.], seed=12345)
        assert res == res2, "{0} != {1}".format(res, res2)
        assert error == error2

    def test_args(self):
        """
        Check that MISER works with arguments.
        """
        func_noargs = lambda x: x**2
        func_args = lambda x, a: a*x**2
        npoints = 50000
        res_noargs, error_noargs = mcmiser(func_noargs, npoints, [0.],[1.], seed=12345)
        res_args, error_args = mcmiser(func_args, npoints, [0.],[1.], seed=12345, args=(1,))
        self.assertEqual(res_noargs, res_args)
        self.assertEqual(error_noargs, error_args)

    def test_args2(self):
        """
        Check that MISER works with arguments (2)
        """
        func_noargs = lambda x: x**2
        func_args = lambda x, a: a*x**2
        npoints = 50000
        res_noargs, error_noargs = mcmiser(func_noargs, npoints, [0.],[1.], seed=12345)
        res_args, error_args = mcmiser(func_args, npoints, [0.],[1.], seed=12345, args=(2,))
        self.assertEqual(2*res_noargs, res_args)
        self.assertEqual(2*error_noargs, error_args)

    def test_zero_volume(self):
        """
        Passing empty integration volume to MISER raises ValueError.
        """
        with self.assertRaises(ValueError):
            mcmiser(lambda x:x**2, 20000, [0.],[0.])
        

if __name__ == '__main__':
    import sys
    run_module_suite(argv=sys.argv)
