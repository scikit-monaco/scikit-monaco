
import numpy as np
import numpy.random
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt
import time
from libc.stdlib cimport malloc, free
from cython.parallel import prange, parallel
import os

cimport _core as core

ctypedef cnp.float64_t DOUBLE

cdef run_integral(object f,int npts, int dim, cnp.ndarray[DOUBLE,ndim=2] pts, 
        double weight, object args):
    cdef :
        double summ_double, sum2_double, first_val_double, npts_float 
        bint ret_object
        object summ_object, sum2_object, first_val_object

    # Do one call to get return type.
    first_val_object = f(pts[-1,:],*args)
    try:
        first_val_double = float(first_val_object)
        ret_object = False
    except TypeError: # object cannot be cast to float
        ret_object = True

    # Run the bulk of the integration.
    if not ret_object:
        if len(args) > 0:
            core.mc_kernel(f,npts-1,dim,pts,args,&summ_double,&sum2_double)
        else:
            core.mc_kernel_noargs(f,npts-1,dim,pts,&summ_double,&sum2_double)
        summ_object = <object>summ_double
        sum2_object = <object>sum2_double
    else:
        if len(args) > 0:
            summ_object, sum2_object = core.mc_kernel_ret_object(
                    f,npts-1,dim,pts,args)
        else:
            summ_object, sum2_object = core.mc_kernel_ret_object_no_args(
                    f,npts-1,dim,pts)

    summ_object += first_val_object
    sum2_object += first_val_object*first_val_object
    summ_object *= weight
    sum2_object *= weight**2
    return summ_object,sum2_object

def integrate_points(f, pts, double weight=1.0, object args=()):
    cdef:
        int dim, npoints
        cnp.ndarray[DOUBLE,ndim=2] points

    if np.rank(pts) == 1:
        dim = 1
        npoints = len(pts)
        points = <cnp.ndarray[DOUBLE,ndim=2]> pts[:,None]
        assert points.shape[0] == npoints and points.shape[1] == 1
    else:
        if not np.rank(pts) == 2:
            raise ValueError(
                    "Pts must be a rank-2 array: (npoints,dim).")
        points = <cnp.ndarray[DOUBLE,ndim=2]> pts
        npoints, dim = np.shape(points)
    return run_integral(f,npoints,dim,points,weight,args)

def integrate_uniform(f,int npoints, xl, xu, args=(),rng=numpy.random,seed=None):
    cdef :
        int dim = len(xl)
        cnp.ndarray[DOUBLE,ndim=2] points
        cnp.ndarray[DOUBLE,ndim=1] xl_a = xl
        cnp.ndarray[DOUBLE,ndim=1] xu_a = xu
        int i

    if npoints < 2:
        raise ValueError("'npoints must be >= 2.")

    rng.seed(seed)

    points = rng.ranf((npoints,dim))
    volume = abs(np.multiply.reduce(xu-xl))
    if volume == 0.:
        raise ValueError("Integration volume is zero.")
    core.generate_points(npoints, dim, &(xl_a[0]), &(xu_a[0]), points)
    return run_integral(f,npoints,dim,points,volume,args)
    
def integrate_importance(f,int npoints, distribution, 
        args=(), rng=numpy.random, seed=None,dist_kwargs={},weight=1.0):
    cdef :
        cnp.ndarray[DOUBLE,ndim=2] points
        int dim

    if npoints < 2:
        raise ValueError("'npoints must be >= 2.")

    rng.seed(seed)

    pts_generated = distribution(size=1,**dist_kwargs)

    #t0 = time.time()
    if np.size(pts_generated) == 1 and np.rank(pts_generated) == 1:
        dim = 1
        points = distribution(size=(npoints,1),**dist_kwargs).\
                reshape((npoints,dim))
    else:
        dim = np.size(pts_generated)
        points = distribution(size=npoints,**dist_kwargs).\
                reshape((npoints,dim))
    #t1 = time.time() ; print "Time taken generating points: ",t1-t0
    summ,sum2 = run_integral(f,npoints,dim,points,weight,args)
    #t2 = time.time(); print "Time taken evaluating function: ",t2-t1
    return summ,sum2


