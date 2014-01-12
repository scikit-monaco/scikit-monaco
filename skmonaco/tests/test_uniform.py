
from __future__ import division

import numpy as np
from numpy.testing import TestCase, assert_almost_equal
from numpy.testing.decorators import slow
from utils import assert_within_tol, run_module_suite

from skmonaco import mcquad

HALF_ROOT_PI = 0.5*np.sqrt(np.pi)

class TestMCQuad(TestCase):
    """
    The expected variance for N integration points is:

    (<f^2> - <f>^2) / N 

    where <.> denotes the integration of function f.
    """

    def setUp(self):
        self.gaussian = lambda x: np.exp(-sum(x**2))

    def calc_volume(self,xl,xu):
        xl = np.array(xl)
        xu = np.array(xu)
        return np.multiply.reduce(abs(xu-xl))


    def run_serial(self,f,npoints,expected_value,expected_variance,**kwargs):
        res, sd = mcquad(f,npoints,nprocs=1,**kwargs)
        volume = self.calc_volume(kwargs["xl"],kwargs["xu"])
        error = volume*np.sqrt(expected_variance/float(npoints))
        assert_within_tol(res,expected_value,3.*max(error,1e-10),
            "Error in <f> in serial run.")
        assert_within_tol(sd,error,0.1*max(error,1e-10),
            "Error in expected error in serial run.")

    def run_parallel(self,f,npoints,expected_value,expected_variance,**kwargs):
        batch_size = npoints/10
        res, sd = mcquad(f,npoints,nprocs=2,batch_size=batch_size,**kwargs)
        volume = self.calc_volume(kwargs["xl"],kwargs["xu"])
        error = volume*np.sqrt(expected_variance/float(npoints))
        assert_within_tol(res,expected_value,3.*max(error,1e-10),
                "Error in <f> in parallel run.")
        assert_within_tol(sd,error,0.1*max(error,1e-10),
                "Error in expected error in parallel run.")

    def run_all(self,f,npoints,expected_value,expected_variance,**kwargs):
        self.run_serial(f,npoints,expected_value,expected_variance,**kwargs)
        self.run_parallel(f,npoints,expected_value,expected_variance,**kwargs)

    def run_check_unseeded_distribution(self,f,ntrials,*args,**kwargs):
        """
        Check that the results returned by integrating f are normally distributed.

        Does not try to seed each trial.
        """
        import scipy.stats
        results, errors = [], []
        for itrial in range(ntrials):
            res, err = mcquad(f,*args,**kwargs)
            results.append(res)
            errors.append(err)
        results = np.array(results).flatten()
        w,p = scipy.stats.shapiro(results)
        self.assertGreater(p,0.1)

    def run_check_seeded_distribution(self,f,ntrials,*args,**kwargs):
        """
        Check that the results returned by integrating f are normally distributed.

        Seeds each trial with the trial number.
        """
        import scipy.stats
        results, errors = [], []
        for itrial in range(ntrials):
            res, err = mcquad(f,*args,seed=itrial,**kwargs)
            results.append(res)
            errors.append(err)
        results = np.array(results).flatten()
        w,p = scipy.stats.shapiro(results)
        self.assertGreater(p,0.1)

    def const(self,x):
        """
        Constant function.

        <f> = 1.0
        <(f - <f>)^2> = 0.0
        """
        return 1.0

    def prod(self,x):
        """
        Product_i x_i.

        If the integral region is between 0 and 1:

        <f> = 1/2^d
        <(f-<f>)^2> = (1/3)^d - 1/2^(d-1)+1
        """
        return np.multiply.reduce(x)

    def prod_variance(self,d):
        """ 
        Variance of the function Product_i x_i as a function of the 
        dimensionality.
        """
        return (1./3.**d) - 0.25**d

    def test_const_1d(self):
        """
        Constant function between 0 and 1.

        Value : 1.
        Variance : 0.
        """
        self.run_all(self.const,2000,1.,0.,xl=[0.],xu=[1.])

    def test_const_1db(self):
        """
        Constant function between -1 and 2.

        Value: 3.
        Variance 0.
        """
        self.run_all(self.const,2000,3.0,0.0,xl=[-1.],xu=[2.])

    def test_const_6d(self):
        """
        Constant function between -1. and 2. in six dimensions.

        Value: 3**6
        Variance: 0.
        """
        self.run_all(self.const,20000,3.0**6,0.0,xl=[-1.]*6,xu=[2.]*6)

    def test_gaussian1d(self):
        pass

    def test_prod1d(self):
        """ f(x) = x between 0 and 1. """
        npoints = 2000
        variance = self.prod_variance(1)
        self.run_all(self.prod,npoints,0.5,variance,xl=[0.],xu=[1.])

    def test_prod1db(self):
        """ f(x) = x between -2 and 1. """
        npoints = 2000
        variance = (1+2**3)/3. - 1.5**2
        self.run_all(self.prod,npoints,-1.5,variance,xl=[-2.],xu=[1.])

    def test_prod2d(self):
        """ f(x,y) = x*y between 0 and 1. """
        npoints = 2000
        variance = self.prod_variance(2)
        self.run_all(self.prod,npoints,0.25,variance,xl=[0.,0.],xu=[1.,1.])

    def test_prod6d(self):
        """ f(x,...) = product_1..6 x_i between 0 and 1. """
        npoints = 50000
        variance = self.prod_variance(6)
        self.run_all(self.prod,npoints,0.5**6,variance,xl=[0.]*6,xu=[1.]*6)

    @slow
    def test_distribution_serial_unseeded(self):
        """
        Check that unseeded integrals are normally distributed (serial).

        Use Shapiro-Wilkes test for normality.
        """
        ntrials = 1000
        npoints = 1e4
        self.run_check_unseeded_distribution(lambda x:x**2,
                ntrials,npoints,[0.],[1.])

    @slow
    def test_distribution_serial_seeded(self):
        """
        Check that seeded integrals are normally distributed (serial).

        Use Shapiro-Wilkes test for normality.
        """
        ntrials = 1000
        npoints = 1e4
        self.run_check_seeded_distribution(lambda x:x**2,
                ntrials,npoints,[0.],[1.])

    @slow
    def test_distribution_parallel_unseeded(self):
        """
        Check that unseeded integrals are normally distributed (parallel).

        Use Shapiro-Wilkes test for normality.
        """
        ntrials = 1000
        npoints = 1e4
        self.run_check_unseeded_distribution(lambda x:x**2,
                ntrials,npoints,[0.],[1.],nprocs=2,batch_size=npoints/10)

    @slow
    def test_distribution_parallel_seeded(self):
        """
        Check that seeded integrals are normally distributed (parallel).

        Use Shapiro-Wilkes test for normality.
        """
        ntrials = 1000
        npoints = 1e4
        self.run_check_seeded_distribution(lambda x:x**2,
                ntrials,npoints,[0.],[1.],nprocs=2,batch_size=npoints/10)

    def test_args(self):
        """
        Test passing args to the function.

        f(x ; a) = a*x
        """
        a = 2.
        func = lambda x, a_: a_*x
        npoints = 2000
        variance = 4.*self.prod_variance(1)
        self.run_all(func,npoints,a*0.5,variance,xl=[0.],xu=[1.],args=(a,))

    def test_ret_arr(self):
        """
        Test an integrand that returns an array.
        """
        func = lambda x: np.array((x**2,x**3))
        npoints = 2000
        (res_sq, res_cb), (sd_sq, sd_cb) = mcquad(func,npoints,[0.],[1.],nprocs=1,
                seed=123456)
        res_sq2, sd_sq2 = mcquad(lambda x: x**2,npoints,[0.],[1.],nprocs=1,seed=123456)
        res_cb2, sd_cb2 = mcquad(lambda x: x**3,npoints,[0.],[1.],nprocs=1,seed=123456)
        assert_almost_equal(res_sq, res_sq2)
        assert_almost_equal(res_cb, res_cb2)
        assert_almost_equal(sd_sq, sd_sq2)
        assert_almost_equal(sd_cb, sd_cb2)

    def test_ret_arr_parallel(self):
        """
        Test an integrand that returns an array: parallel implementation.
        """
        func = lambda x: np.array((x**2,x**3))
        npoints = 5000
        nprocs = 2
        (res_sq, res_cb), (sd_sq, sd_cb) = mcquad(func,npoints,[0.],[1.],nprocs=nprocs,
                seed=123456)
        res_sq2, sd_sq2 = mcquad(lambda x: x**2,npoints,[0.],[1.],
                nprocs=nprocs,seed=123456)
        res_cb2, sd_cb2 = mcquad(lambda x: x**3,npoints,[0.],[1.],
                nprocs=nprocs,seed=123456)
        assert_almost_equal(res_sq, res_sq2)
        assert_almost_equal(res_cb, res_cb2)
        assert_almost_equal(sd_sq, sd_sq2)
        assert_almost_equal(sd_cb, sd_cb2)

    def test_ret_arr_args(self):
        """
        Test an integrand that returns an array with an argument.
        """
        func = lambda x, a,b : np.array((a*x**2,b*x**3))
        npoints = 2000
        aval, bval = 4.,5.
        (res_sq, res_cb), (sd_sq, sd_cb) = mcquad(func,npoints,[0.],[1.],nprocs=1,
                seed=123456,args=(aval,bval))
        res_sq2, sd_sq2 = mcquad(lambda x,a: a*x**2,npoints,[0.],[1.],nprocs=1,
                seed=123456,args=(aval,))
        res_cb2, sd_cb2 = mcquad(lambda x,b: b*x**3,npoints,[0.],[1.],nprocs=1,
                seed=123456,args=(bval,))
        assert_almost_equal(res_sq, res_sq2)
        assert_almost_equal(res_cb, res_cb2)
        assert_almost_equal(sd_sq, sd_sq2)
        assert_almost_equal(sd_cb, sd_cb2)
        
    def test_wrong_xl(self):
        """
        Raise a ValueError if len(xl) != len(xu).
        """
        with self.assertRaises(ValueError):
            mcquad(self.const,2000,xl=[0.,0.],xu=[1.])

    def test_wrong_nprocs(self):
        """
        Raise a ValueError if nprocs < 1
        """
        with self.assertRaises(ValueError):
            mcquad(self.const,2000,xl=[0.],xu=[1.],nprocs=-1)

    def test_wrong_npoints(self):
        """
        Raise a ValueError if npoints < 2.
        """
        with self.assertRaises(ValueError):
            mcquad(self.const,0,xl=[0.],xu=[1.])

    def test_seed(self):
        """
        Test same seed -> same result.
        """
        npoints = 50000
        res,error = mcquad(lambda x:x**2,npoints,xl=[0.],xu=[1.],seed=[1234,5678])
        res2, error2 = mcquad(lambda x:x**2,npoints,xl=[0.],xu=[1.],seed=[1234,5678])
        assert res == res2
        assert error == error2

    def test_seed_different(self):
        """
        Test different seed -> different result.
        """
        npoints = 50000
        res,error = mcquad(lambda x: x**2,npoints,xl=[0.],xu=[1.],seed=[1235,5678])
        res2, error2 = mcquad(lambda x: x**2,npoints,xl=[0.],xu=[1.],seed=[1234,5678])
        assert res != res2
        assert error != error2


if __name__ == '__main__':
    # Command line arguments are passed directly to 'nose'.
    # Eg. run '$ python test_uniform.py --eval-attr="not slow"' to 
    # avoid running the "slow" tests.
    import sys
    run_module_suite(argv=sys.argv)

