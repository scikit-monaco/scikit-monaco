
import numpy as np
from numpy.testing import TestCase, assert_almost_equal
import numpy.random
from utils import assert_within_tol, run_module_suite

from skmonaco import integrate_from_points

class TestIntegrateFromPoints(TestCase):

    def setUp(self):
        pass

    def calc_res_err(self,f,points):
        npoints = len(points)
        res = sum(map(f,points))/float(npoints)
        err = np.sqrt(
                sum(map(lambda pt: f(pt)**2,points))
                -npoints*res**2)/npoints
        return res,err

    def run_all(self,f,points,serial_only=False,**kwargs):
        expected_res,expected_err = self.calc_res_err(f,points)
        # serial run
        res, err = integrate_from_points(f,points,nprocs=1,**kwargs)
        assert_within_tol(res,expected_res,1e-10)
        assert_within_tol(err,expected_err,1e-10)
        # parallel run
        if not serial_only:
            res, err = integrate_from_points(f,points,
                    nprocs=2,batch_size=len(points)/10,**kwargs)
            assert_within_tol(res,expected_res,1e-10)
            assert_within_tol(err,expected_err,1e-10)

    def test_const_1d(self):
        """
        Constant function in one dimension.
        """
        points = np.array((1.,2.,3.))
        res, err = integrate_from_points(lambda x: 1.0,points)
        assert (res,err) == (1.0,0.0)

    def test_const_1db(self):
        """
        Constant function in one dimension -- alternate points representation.
        """
        points = np.array((1.,2.,3.)).reshape((3,1))
        res, err = integrate_from_points(lambda x: 1.0,points)
        assert (res,err) == (1.0,0.0)

    def test_x_squared_1d(self):
        """ x**2 in 1 dimension. """
        points = np.array((1.,2.,3.)).reshape((3,1))
        self.run_all(lambda x: x**2, points,serial_only=True)

    def test_x_squared_2d(self):
        """ x**2 + 3*y**2 """
        points = np.array([(1.,2.),(3.,5.),(6.,7.)])
        self.run_all(lambda xs: xs[0]**2 + 3.*xs[1]**2,points,
                serial_only=True)

    def test_x_squared_2db(self):
        """ x**2 + 3*y**2, both serial and parallel. """
        npoints = 20000
        points = numpy.random.ranf(size=2*npoints).reshape(npoints,2)
        self.run_all(lambda xs:xs[0]**2 + 3*xs[1]**2,points)

    def test_args(self):
        """ x**2 + a*y**2, passing a as an argument. """
        npoints = 1000
        points = numpy.random.ranf(size=2*npoints).reshape(npoints,2)
        aval = 3.0
        f = lambda xs,a: xs[0]**2 + a*xs[1]**2
        expected_res, expected_err = self.calc_res_err(
                lambda xs:xs[0]**2 + aval*xs[1]**2,points)
        res_serial, err_serial = integrate_from_points(f,points,
                args=(aval,),nprocs=1)
        assert_within_tol(res_serial,expected_res,1e-10)
        assert_within_tol(err_serial,expected_err,1e-10)
        res_parallel, err_parallel = integrate_from_points(f,points,
                args=(aval,),nprocs=2,batch_size=npoints/2)
        assert_within_tol(res_parallel,expected_res,1e-10)
        assert_within_tol(err_parallel,expected_err,1e-10)

    def test_ret_arr(self):
        """ Test an integrand that returns an array """
        npoints = 2000
        points = numpy.random.ranf(size=npoints)
        func = lambda x: np.array((x**2,x**3))
        (res_sq, res_cb), (sd_sq, sd_cb) = integrate_from_points(func,points,nprocs=1)
        res_sq2, sd_sq2 = integrate_from_points(lambda x: x**2,points,nprocs=1)
        res_cb2, sd_cb2 = integrate_from_points(lambda x: x**3,points,nprocs=1) 
        assert_almost_equal(res_sq, res_sq2,decimal=12)
        assert_almost_equal(res_cb, res_cb2,decimal=12)
        assert_almost_equal(sd_sq, sd_sq2,decimal=12)
        assert_almost_equal(sd_cb, sd_cb2,decimal=12)

    def test_ret_arr_args(self):
        """
        Test an integrand that returns an array with an argument.
        """
        npoints = 1000
        points = numpy.random.ranf(size=npoints)
        func = lambda x,a,b: np.array((a*x**2,b*x**3))
        aval, bval = 4.,5.
        (res_sq, res_cb), (sd_sq, sd_cb) = integrate_from_points(func,points,nprocs=1,
                args=(aval,bval))
        res_sq2, sd_sq2 = integrate_from_points(lambda x,a: a*x**2,points,nprocs=1,
                args=(aval,))
        res_cb2, sd_cb2 = integrate_from_points(lambda x,b: b*x**3,points,nprocs=1,
                args=(bval,))
        assert_almost_equal(res_sq, res_sq2,decimal=12)
        assert_almost_equal(res_cb, res_cb2,decimal=12)
        assert_almost_equal(sd_sq, sd_sq2,decimal=12)
        assert_almost_equal(sd_cb, sd_cb2,decimal=12)

    def test_weight(self):
        """ a*x**2, using a weight. """
        npoints = 1000
        points = numpy.random.ranf(size=npoints).reshape(npoints,1)
        aval = 3.0
        f = lambda x: x**2
        expected_res, expected_err = self.calc_res_err(
                lambda x:aval*x**2,points)
        res_serial, err_serial = integrate_from_points(f,points,
                nprocs=1,weight=aval)
        assert_within_tol(res_serial,expected_res,1e-10)
        assert_within_tol(err_serial,expected_err,1e-10)
        res_parallel, err_parallel = integrate_from_points(f,points,
                nprocs=2,batch_size=npoints/2,weight=aval)
        assert_within_tol(res_parallel,expected_res,1e-10)
        assert_within_tol(err_parallel,expected_err,1e-10)
        
    
if __name__ == '__main__':
    import sys
    run_module_suite(argv=sys.argv)
