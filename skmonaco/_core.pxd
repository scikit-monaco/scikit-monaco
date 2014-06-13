
cimport numpy as cnp

ctypedef cnp.float64_t DOUBLE

# Transform from points in U[0,1]^dim to 
# points uniformly distributed in a volume
# set by xl and xu.
cdef bint generate_points(int npoints, int dim, 
        double* xl, double* xu, cnp.ndarray[DOUBLE,ndim=2] pts) except 0

# Kernels

cdef bint mc_kernel(object f, int npts, int dim, cnp.ndarray[DOUBLE,ndim=2] pts,
        object args, double* summ, double* sum2) except 0

cdef bint mc_kernel_noargs(object f, int npts, int dim, cnp.ndarray[DOUBLE,ndim=2] pts,
        double* summ, double* sum2) except 0

cdef object mc_kernel_ret_object_no_args(object f, int npts, int dim, 
        cnp.ndarray[DOUBLE,ndim=2] pts)

cdef object mc_kernel_ret_object(object f, int npts, int dim, 
        cnp.ndarray[DOUBLE,ndim=2] pts, object args)
