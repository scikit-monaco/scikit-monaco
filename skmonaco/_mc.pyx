
import numpy as np
import numpy.random
cimport numpy as np
cimport cython
from libc.math cimport sqrt
import time
from libc.stdlib cimport malloc, free
from cython.parallel import prange, parallel
import os

ctypedef np.float64_t DOUBLE

cdef void mc_kernel(object f, int npts, int dim, np.ndarray[DOUBLE,ndim=2] pts, 
        object args, double* summ, double* sum2):
    cdef :
        int ipt,i
        double val
        np.ndarray[DOUBLE,ndim=1] pt = np.empty((dim,))
        double sum_tmp = 0.0, sum2_tmp = 0.0
    for ipt in range(npts):
        for i in range(dim):
            pt[i] = pts[ipt,i]
        val = f(pt,*args)
        sum_tmp += val
        sum2_tmp += val*val
    summ[0] = sum_tmp
    sum2[0] = sum2_tmp

cdef mc_kernel_noargs(object f, int npts, int dim, np.ndarray[DOUBLE,ndim=2] pts,
        double* summ, double* sum2):
    cdef :
        int ipt,i
        double val
        np.ndarray[DOUBLE,ndim=1] pt = np.empty((dim,))
        double sum_tmp = 0.0, sum2_tmp = 0.0
    for ipt in range(npts):
        for i in range(dim):
            pt[i] = pts[ipt,i]
        val = f(pt)
        sum_tmp += val
        sum2_tmp += val*val
    summ[0] = sum_tmp
    sum2[0] = sum2_tmp

cdef mc_kernel_ret_object_no_args(object f, int npts, int dim, 
        np.ndarray[DOUBLE,ndim=2] pts):
    cdef :
        int ipt, i
        object val,summ,sum2
        np.ndarray[DOUBLE,ndim=1] pt = np.empty((dim,))
    summ = 0.0
    sum2 = 0.0
    for ipt in range(npts):
        for i in range(dim):
            pt[i] = pts[ipt,i]
        val = f(pt)
        summ += val
        sum2 += val*val
    return summ,sum2

cdef mc_kernel_ret_object(object f, int npts, int dim, 
        np.ndarray[DOUBLE,ndim=2] pts, object args):
    cdef :
        int ipt, i
        object val,summ,sum2
        np.ndarray[DOUBLE,ndim=1] pt = np.empty((dim,))
    summ = 0.0
    sum2 = 0.0
    for ipt in range(npts):
        for i in range(dim):
            pt[i] = pts[ipt,i]
        val = f(pt,*args)
        summ += val
        sum2 += val*val
    return summ,sum2

cdef void generate_points(int npoints, int dim, 
        np.ndarray[DOUBLE,ndim=1] xl, 
        np.ndarray[DOUBLE,ndim=1] xu, 
        np.ndarray[DOUBLE,ndim=2] pts):
    cdef int ipt, idim
    for ipt in range(npoints):
        for idim in range(dim):
            pts[ipt,idim] = xl[idim] + (xu[idim]-xl[idim])*pts[ipt,idim]

cdef run_integral(object f,int npts, int dim, np.ndarray[DOUBLE,ndim=2] pts, 
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
            mc_kernel(f,npts-1,dim,pts,args,&summ_double,&sum2_double)
        else:
            mc_kernel_noargs(f,npts-1,dim,pts,&summ_double,&sum2_double)
        summ_object = <object>summ_double
        sum2_object = <object>sum2_double
    else:
        if len(args) > 0:
            summ_object, sum2_object = mc_kernel_ret_object(
                    f,npts-1,dim,pts,args)
        else:
            summ_object, sum2_object = mc_kernel_ret_object_no_args(
                    f,npts-1,dim,pts)

    summ_object += first_val_object
    sum2_object += first_val_object*first_val_object
    summ_object *= weight
    sum2_object *= weight**2
    return summ_object,sum2_object

def integrate_points(f, pts, double weight=1.0, object args=()):
    cdef:
        int dim, npoints
        np.ndarray[DOUBLE,ndim=2] points

    if np.rank(pts) == 1:
        dim = 1
        npoints = len(pts)
        points = <np.ndarray[DOUBLE,ndim=2]> pts[:,None]
        assert points.shape[0] == npoints and points.shape[1] == 1
    else:
        if not np.rank(pts) == 2:
            raise ValueError(
                    "Pts must be a rank-2 array: (npoints,dim).")
        points = <np.ndarray[DOUBLE,ndim=2]> pts
        npoints, dim = np.shape(points)
    return run_integral(f,npoints,dim,points,weight,args)

def integrate_uniform(f,int npoints, xl, xu, args=(),rng=numpy.random,seed=None):
    cdef :
        int dim = len(xl)
        np.ndarray[DOUBLE,ndim=2] points
        np.ndarray[DOUBLE,ndim=1] xl_a = xl
        np.ndarray[DOUBLE,ndim=1] xu_a = xu
        int i

    if npoints < 2:
        raise ValueError("'npoints must be >= 2.")

    rng.seed(seed)

    points = rng.ranf((npoints,dim))
    volume = abs(np.multiply.reduce(xu-xl))
    generate_points(npoints, dim, xl_a, xu_a, points)
    return run_integral(f,npoints,dim,points,volume,args)
    
def integrate_importance(f,int npoints, distribution, 
        args=(), rng=numpy.random, seed=None,dist_kwargs={},weight=1.0):
    cdef :
        np.ndarray[DOUBLE,ndim=2] points
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


