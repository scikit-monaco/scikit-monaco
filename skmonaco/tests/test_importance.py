
import numpy as np
from numpy.testing import TestCase, assert_almost_equal
from numpy.testing.decorators import slow
from numpy.random import exponential,uniform
from utils import assert_within_tol, run_module_suite

from skmonaco import mcimport

class TestMCImport(TestCase):
    """
    The expected variance for N integration points is,
    if the function is f and the probability distribution is g:

    sqrt{int f^2 * g dx - (int f*g dx)^2 / N}
    """

    def setUp(self):
        pass

    def exp_integral(self,d):
        return (1.-np.exp(-1))**d

    def exp_variance(self,d):
        return self.exp_integral(d) - self.exp_integral(d)**2

    def run_serial(self,f,npoints,distribution,expected_value,expected_variance,**kwargs):
        res,sd = mcimport(f,npoints,distribution,nprocs=1,**kwargs)
        error = np.sqrt(expected_variance/float(npoints))
        assert_within_tol(res,expected_value,3.*max(error,1e-10),
            "Error in <f> in serial run.")
        assert_within_tol(sd,error,0.1*max(error,1e-10),
            "Error in expected error in serial run.")

    def run_parallel(self,f,npoints,distribution,expected_value,expected_variance,**kwargs):
        res,sd = mcimport(f,npoints,distribution,nprocs=2,**kwargs)
        error = np.sqrt(expected_variance/float(npoints))
        assert_within_tol(res,expected_value,3.*max(error,1e-10),
            "Error in <f> in parallel run.")
        assert_within_tol(sd,error,0.1*max(error,1e-10),
            "Error in expected error in parallel run.")

    def run_all(self,f,npoints,distribution,expected_value,expected_variance,**kwargs):
        self.run_serial(f,npoints,distribution,expected_value,expected_variance,**kwargs)
        self.run_parallel(f,npoints,distribution,expected_value,expected_variance,**kwargs)

    def run_check_unseeded_distribution(self,f,ntrials,*args,**kwargs):
        """
        Check that the results returned by integrating f are normally distributed.

        Does not try to seed each trial.
        """
        import scipy.stats
        results, errors = [], []
        for itrial in range(ntrials):
            res, err = mcimport(f,*args,**kwargs)
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
            res, err = mcimport(f,*args,seed=itrial,**kwargs)
            results.append(res)
            errors.append(err)
        results = np.array(results).flatten()
        w,p = scipy.stats.shapiro(results)
        self.assertGreater(p,0.1)

    def test_exp_1d(self):
        """
        e^-x for x = 0..1 with g(x) = e^-x
        """
        npoints = 2000
        self.run_all(lambda x:x<1.0,npoints,exponential,
                self.exp_integral(1),self.exp_variance(1))

    def test_exp_1db(self):
        """
        e^-x for x = 0..1 with g(x) = e^-x : alternate dist formulation
        """
        npoints = 2000
        self.run_all(lambda x:x<1.0,npoints,lambda size: exponential(size=(size,1)),
                self.exp_integral(1),self.exp_variance(1))

    def test_exp_2d(self):
        """
        e^-(x+y) for x,y = 0..1 with g(x) = e^-(x+y)
        """
        npoints = 2000
        self.run_all(lambda xs:xs[0]<1.0 and xs[1]<1.0,npoints,
                lambda size:(exponential(size=(size,2))),
                self.exp_integral(2),self.exp_variance(2))

    def test_exp_6d(self):
        """
        e^-(sum(x)) for x = 0..1 with g(x) = e^-(sum(x)) for d=6.
        """
        npoints = 10000
        self.run_all(lambda x:np.all(x<1.0),npoints,
                lambda size:(exponential(size=(size,6))),
                self.exp_integral(6),self.exp_variance(6))

    def test_mixed_distributions(self):
        """
        exp^(-x)*y**2 for x,y = 0..1 with g(x) ~ Exp and g(y) ~ U[0,1]
        """
        npoints = 2000
        def dist(size):
            xs = exponential(size=size)
            ys = uniform(size=size)
            return np.array((xs,ys)).T
        self.run_all(lambda xs:xs[1]**2*(xs[0]<1.),npoints,dist,
                self.exp_integral(1)/3.,
                self.exp_integral(1)/5.-(self.exp_integral(1)**2)/9.)

    @slow
    def test_distribution_serial_unseeded(self):
        """
        Check that unseeded integrals are normally distributed (serial).
        
        Use Shapiro-Wilkes test for normality.
        """
        ntrials = 1000
        npoints = 1e4
        self.run_check_unseeded_distribution(lambda x: x<1.0,
                ntrials,npoints,exponential)

    @slow
    def test_distribution_serial_seeded(self):
        """
        Check that seeded integrals are normally distributed (serial).
        
        Use Shapiro-Wilkes test for normality.
        """
        ntrials = 1000
        npoints = 1e4
        self.run_check_seeded_distribution(lambda x: x<1.0,
                ntrials,npoints,exponential)

    @slow
    def test_distribution_parallel_unseeded(self):
        """
        Check that unseeded integrals are normally distributed (parallel).
        
        Use Shapiro-Wilkes test for normality.
        """
        ntrials = 1000
        npoints = 1e4
        self.run_check_unseeded_distribution(lambda x: x<1.0,
                ntrials,npoints,exponential,nprocs=2,batch_size=npoints/10)

    @slow
    def test_distribution_parallel_seeded(self):
        """
        Check that seeded integrals are normally distributed (parallel).
        
        Use Shapiro-Wilkes test for normality.
        """
        ntrials = 1000
        npoints = 1e4
        self.run_check_seeded_distribution(lambda x: x<1.0,
                ntrials,npoints,exponential,nprocs=2,batch_size=npoints/10)

    def test_weight(self):
        """
        Weight 'exp^(-x)' by 2.0.
        """
        npoints = 2000
        self.run_all(lambda x: x<1.0,npoints,exponential,
                2.0*self.exp_integral(1),4.0*self.exp_variance(1),weight=2.0)

    def test_args(self):
        """
        a*exp(-x) where a is an "arg" passed to f.
        """
        npoints = 2000
        aval = 2.0
        self.run_all(lambda x,a: a*(x<1.0), npoints,exponential,
                aval*self.exp_integral(1),aval**2*self.exp_variance(1),
                args=(aval,))

    def test_dist_kwargs(self):
        """
        exp(-x/c) where c is an "arg" passed to g.
        """
        npoints = 2000
        cval = 2.0
        exp_integral = cval*(1.-np.exp(-1./cval))
        self.run_all(lambda x:cval*(x<1.0), npoints, exponential,
                exp_integral,cval*exp_integral-exp_integral**2,
                dist_kwargs=dict(scale=cval))

    def test_ret_arr(self):
        """
        Test an integrand that returns an array.
        """
        func = lambda x: np.array((x**2,x**3))
        npoints = 2000
        (res_sq, res_cb), (sd_sq, sd_cb) = mcimport(func,npoints,distribution=exponential,
                nprocs=1, seed=123456)
        res_sq2, sd_sq2 = mcimport(lambda x: x**2,npoints,distribution=exponential,
                nprocs=1,seed=123456)
        res_cb2, sd_cb2 = mcimport(lambda x: x**3,npoints,distribution=exponential,
                nprocs=1,seed=123456)
        assert_almost_equal(res_sq, res_sq2)
        assert_almost_equal(res_cb, res_cb2)
        assert_almost_equal(sd_sq, sd_sq2)
        assert_almost_equal(sd_sq, sd_sq2)

    def test_ret_arr_args(self):
        """
        Test an integrand that returns an array with an argument.
        """
        func = lambda x, a,b : np.array((a*x**2,b*x**3))
        npoints = 2000
        aval, bval = 4.,5.
        (res_sq, res_cb), (sd_sq, sd_cb) = mcimport(func,npoints,distribution=exponential,
                nprocs=1,seed=123456,args=(aval,bval))
        res_sq2, sd_sq2 = mcimport(lambda x,a: a*x**2,npoints,distribution=exponential,
                nprocs=1,seed=123456,args=(aval,))
        res_cb2, sd_cb2 = mcimport(lambda x,b: b*x**3,npoints,distribution=exponential,
                nprocs=1,seed=123456,args=(bval,))
        assert_almost_equal(res_sq, res_sq2)
        assert_almost_equal(res_cb, res_cb2)
        assert_almost_equal(sd_sq, sd_sq2)
        assert_almost_equal(sd_sq, sd_sq2)

    def test_seed(self):
        """
        Test same seed -> same result.
        """
        npoints = 50000
        res,error = mcimport(lambda x: x<1.0,npoints,exponential,seed=[1234,5678])
        res2, error2 = mcimport(lambda x: x<1.0,npoints,exponential,seed=[1234,5678])
        assert res == res2
        assert error == error2

    def test_seed_different(self):
        """
        Test different seed -> different result.
        """
        npoints = 50000
        res,error = mcimport(lambda x: x<1.0,npoints,exponential,seed=[1234,5678])
        res2, error2 = mcimport(lambda x: x<1.0,npoints,exponential,seed=[1235,5678])
        assert res != res2
        assert error != error2
                

if __name__ == '__main__':
    import sys
    run_module_suite(argv=sys.argv)
