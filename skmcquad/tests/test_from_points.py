
import numpy as np
from numpy.testing import TestCase, run_module_suite, build_err_msg
import numpy.random

from skmcquad import integrate_from_points

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
        self.run_all(lambda (x,y): x**2 + 3.*y**2,points,
                serial_only=True)

    def test_x_squared_2db(self):
        """ x**2 + 3*y**2, both serial and parallel. """
        npoints = 20000
        points = numpy.random.ranf(size=2*npoints).reshape(npoints,2)
        self.run_all(lambda (x,y):x**2 + 3*y**2,points)

    def test_args(self):
        """ x**2 + a*y**2, passing a as an argument. """
        npoints = 1000
        points = numpy.random.ranf(size=2*npoints).reshape(npoints,2)
        aval = 3.0
        f = lambda (x,y),a: x**2 + a*y**2
        expected_res, expected_err = self.calc_res_err(
                lambda (x,y):x**2 + aval*y**2,points)
        res_serial, err_serial = integrate_from_points(f,points,
                args=(aval,),nprocs=1)
        assert_within_tol(res_serial,expected_res,1e-10)
        assert_within_tol(err_serial,expected_err,1e-10)
        res_parallel, err_parallel = integrate_from_points(f,points,
                args=(aval,),nprocs=2,batch_size=npoints/2)
        assert_within_tol(res_parallel,expected_res,1e-10)
        assert_within_tol(err_parallel,expected_err,1e-10)

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
        
    
#FIXME refactor
def within_tol(a,b,tol):
    return np.abs(a-b).max() < tol

def assert_within_tol(a,b,tol,err_msg=""):
    if not within_tol(a,b,tol):
        msg = build_err_msg((a,b),err_msg)
        raise AssertionError(msg)

if __name__ == '__main__':
    run_module_suite()
