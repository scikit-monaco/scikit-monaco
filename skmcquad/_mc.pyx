
import numpy as np
import numpy.random
cimport numpy as np
cimport cython
import time
from libc.stdlib cimport malloc, free
from cython.parallel import prange, parallel
import os

ctypedef np.float64_t DOUBLE

cdef mc_kernel(object f, int npts, int dim, np.ndarray[DOUBLE,ndim=2] pts):
    cdef double summ = 0.0
    cdef double sum2 = 0.0
    cdef int ipt,i
    cdef double val
    cdef np.ndarray[DOUBLE,ndim=1] pt = np.empty((dim,))
    for ipt in range(npts):
        for i in range(dim):
            pt[i] = pts[ipt,i]
        val = f(pt)
        summ += val
        sum2 += val*val
    return summ, sum2

cdef void generate_points(int npoints, int dim, 
        np.ndarray[DOUBLE,ndim=1] xl, 
        np.ndarray[DOUBLE,ndim=1] xu, 
        np.ndarray[DOUBLE,ndim=2] pts):
    cdef int ipt, idim
    for ipt in range(npoints):
        for idim in range(dim):
            pts[ipt,idim] = xl[idim] + (xu[idim]-xl[idim])*pts[ipt,idim]


def integrate_kernel(f,int npoints, xl, xu, rng=numpy.random,seed=None):
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
    summ, sum2 = mc_kernel(f,npoints,dim,points)
    summ *= volume
    sum2 *= volume**2
    average = summ/float(npoints)
    return average, np.sqrt(sum2-summ**2/float(npoints))/float(npoints)
    
