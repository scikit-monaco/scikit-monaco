
cimport numpy as cnp

ctypedef cnp.float64_t DOUBLE

cdef void mc_kernel(object f, int npts, int dim, cnp.ndarray[DOUBLE,ndim=2] pts,
        object args, double* summ, double* sum2)

cdef mc_kernel_noargs(object f, int npts, int dim, cnp.ndarray[DOUBLE,ndim=2] pts,
        double* summ, double* sum2)

cdef object mc_kernel_ret_object_no_args(object f, int npts, int dim, 
        cnp.ndarray[DOUBLE,ndim=2] pts)

cdef object mc_kernel_ret_object(object f, int npts, int dim, 
        cnp.ndarray[DOUBLE,ndim=2] pts, object args)
