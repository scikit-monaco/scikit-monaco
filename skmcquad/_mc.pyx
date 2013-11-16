
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
        double summ, sum2, npts_float, average, sd
    if len(args) > 0:
        mc_kernel(f,npts,dim,pts,args,&summ,&sum2)
    else:
        mc_kernel_noargs(f,npts,dim,pts,&summ,&sum2)
    summ *= weight
    sum2 *= weight**2
    npts_float = <double>npts
    average = summ/npts_float
    sd = sqrt(sum2-summ**2/npts_float)/npts_float
    return average, sd

def integrate_uniform(f,int npoints, xl, xu, args=(),rng=numpy.random,seed=None):
    cdef :
        int dim = len(xl)
        np.ndarray[DOUBLE,ndim=2] points
        np.ndarray[DOUBLE,ndim=1] xl_a = xl
        np.ndarray[DOUBLE,ndim=1] xu_a = xu
        int i

    if npoints < 2:
        raise ValueError("'npoints must be >= 2.")

    if seed is None:
        seed = [time.time()+os.getpid()]
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

    pts_generated = distribution(size=1,**dist_kwargs)

    if seed is None:
        seed = [time.time()+os.getpid()]
    rng.seed(seed)

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
    av,sd = run_integral(f,npoints,dim,points,weight,args)
    #t2 = time.time(); print "Time taken evaluating function: ",t2-t1
    return av,sd


