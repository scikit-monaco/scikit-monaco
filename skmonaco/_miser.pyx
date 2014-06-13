
from libc.stdlib cimport malloc, free
from libc.math cimport fmax,fmin,pow
cimport numpy as cnp
import numpy as np

from cython.view cimport array as carray

cimport _core as core

ctypedef cnp.float64_t DOUBLE

cdef double TINY = 1e-30
cdef double BIG = 1e30

ctypedef struct MiserParams:
    int MNBS    # if fewer points than MNBS: run straight MC.
    double PFAC # fraction of points to use in bisection
    int MNPT    # minimum number of points in bissection.
    double exponent


# KERNELS for calculating the maximum and minimum in 
# each region.

cdef bint miser_get_max_min_noargs(object f, int npoints, int ndims, 
        cnp.ndarray[DOUBLE,ndim=2] points,
        double* xmid, double* maxl, double* minl, double* maxu, double* minu) except 0:
    cdef:
        int ipt
        int idim
        double fval

    for ipt in range(npoints):
        fval = f(points[ipt,:])
        for idim in range(ndims):
            if points[ipt,idim] < xmid[idim]:
                maxl[idim] = fmax(fval,maxl[idim])
                minl[idim] = fmin(fval,minl[idim])
            else:
                maxu[idim] = fmax(fval,maxu[idim])
                minu[idim] = fmin(fval,minu[idim])
    return 1

cdef bint miser_get_max_min_args(object f, int npoints, int ndims,
        cnp.ndarray[DOUBLE,ndim=2] points,
        double* xmid, double* maxl, double* minl, double* maxu, double* minu, object args) except 0:
    cdef:
        int ipt
        int idim
        double fval

    for ipt in range(npoints):
        fval = f(points[ipt,:],*args)
        for idim in range(ndims):
            if points[ipt,idim] < xmid[idim]:
                maxl[idim] = fmax(fval,maxl[idim])
                minl[idim] = fmin(fval,minl[idim])
            else:
                maxu[idim] = fmax(fval,maxu[idim])
                minu[idim] = fmin(fval,minu[idim])
    return 1



cdef bint miser_kernel(object f, object ranf, int npoints, int ndims, 
        double* xl, double* xu, MiserParams* params, int has_args, object args, 
        double* ave, double* var) except 0:
    cdef:
        int idim,ipt,idim_star,npre,npointsl,npointsu
        double summ, sum2, fval,sigl,sigu,sigl_star,sigu_star,npoints_float
        double diffl, diffu,sumdiff,avel,aveu,varl,varu,fracl,fracr
        double* xmid,*maxl,*minl,*maxu,*minu,*newxl,*newxu
        cnp.ndarray[DOUBLE,ndim=2] points

    if npoints < params.MNBS:
        # straight mc
        points = ranf(size=(npoints,ndims))
        core.generate_points(npoints,ndims,xl,xu,points)
        if has_args:
            core.mc_kernel(f, npoints, ndims, points, args, &summ, &sum2)
        else:
            core.mc_kernel_noargs(f,npoints,ndims,points,&summ,&sum2)
        npoints_float = <double>npoints
        ave[0] = summ/npoints_float
        var[0] = (sum2-summ**2/npoints_float)/(npoints_float**2)

    else:
        xmid = <double*>malloc(ndims*sizeof(double))
        if xmid==NULL:
            raise MemoryError
        maxl = <double*>malloc(ndims*sizeof(double))
        if maxl==NULL:
            raise MemoryError
        minl = <double*>malloc(ndims*sizeof(double))
        if minl==NULL:
            raise MemoryError
        maxu = <double*>malloc(ndims*sizeof(double))
        if maxu==NULL:
            raise MemoryError
        minu = <double*>malloc(ndims*sizeof(double))
        if minu==NULL:
            raise MemoryError
        newxl = <double*>malloc(ndims*sizeof(double))
        if newxl==NULL:
            raise MemoryError
        newxu = <double*>malloc(ndims*sizeof(double))
        if newxu==NULL:
            raise MemoryError

        for idim in range(ndims):
            xmid[idim] = 0.5*(xl[idim]+xu[idim])

        npre = <int>fmax(params.PFAC*npoints,<double>params.MNPT)
        points = ranf(size=(npre,ndims))
        core.generate_points(npre,ndims,xl,xu,points)

        # find maximum and minimum for each sub-region
        for idim in range(ndims):
            maxl[idim] = -BIG
            minl[idim] = BIG
            maxu[idim] = -BIG
            minu[idim] = BIG

        if has_args:
            miser_get_max_min_args(f, npre, ndims, points,
                    xmid, maxl, minl, maxu, minu, args)
        else:
            miser_get_max_min_noargs(f, npre, ndims, points,
                    xmid, maxl, minl, maxu, minu)

        sum_star = BIG
        idim_star = -1

        # find sub-region with smallest difference 
        for idim in range(ndims):
            if maxl[idim] > minl[idim] and maxu[idim] > minu[idim]:
                diffl = maxl[idim]-minl[idim]
                diffu = maxu[idim]-minu[idim]
                sumdiff = diffl+diffu
                if sumdiff < sum_star:
                    sum_star = sumdiff
                    idim_star = idim
                    diffl_star = diffl
                    diffu_star = diffu

        if idim_star == -1:
            # Not enough points. Choose direction at random.
            idim_star = <int>(ranf(1)*ndims)
            sigl = 1.0
            sigr = 1.0
        else:
            sigl = fmax(TINY,pow(diffl_star,params.exponent))
            sigu = fmax(TINY,pow(diffu_star,params.exponent))

        for idim in range(ndims):
            newxu[idim] = xu[idim]
            newxl[idim] = xl[idim]
        newxu[idim_star] = xmid[idim_star]
        newxl[idim_star] = xmid[idim_star]

        npointsl = int(<double>params.MNPT+
                <double>(npoints-npre-2*params.MNPT)*sigl/(sigl+sigu))
        npointsu = npoints-npre-npointsl

        # Free temporary arrays
        free(xmid)
        free(maxl)
        free(minl)
        free(maxu)
        free(minu)

        # Recurse
        miser_kernel(f,ranf,npointsl,ndims,xl,newxu,params,has_args,args,
                &avel,&varl)
        miser_kernel(f,ranf,npointsu,ndims,newxl,xu,params,has_args,args,
                &aveu,&varu)

        free(newxl)
        free(newxu)
        
        ave[0] = 0.5*(avel+aveu)
        var[0] = 0.25*(varl+varu)
    return 1


def integrate_miser(f,npoints,xl,xu,args=(),
        min_bisect=100,pre_frac=0.1,exponent=2./3.,rng=None,seed=None):
    cdef MiserParams params
    cdef double ave, var
    cdef cnp.ndarray[DOUBLE] xl_tmp, xu_tmp
    params.MNBS = min_bisect
    params.PFAC = pre_frac
    params.MNPT = min_bisect/4.
    params.exponent = exponent
    if npoints < 2:
        raise ValueError("'npoints must be >= 2.'")
    if rng is None:
        import numpy.random
        rng = numpy.random
    if seed is None:
        import os,time
        seed = [time.time()+os.getpid()]
    rng.seed(seed)
    ranf = rng.ranf
    xl_tmp = np.array(xl)
    xu_tmp = np.array(xu)
    volume = np.abs(np.multiply.reduce(xu_tmp-xl_tmp))
    if volume == 0.:
        raise ValueError("Integration volume is zero.")
    assert len(xl_tmp) == len(xu_tmp)
    miser_kernel(f,ranf,npoints,len(xl_tmp),<double*>xl_tmp.data,<double*>xu_tmp.data,
            &params, 1 if len(args) else 0, args, &ave, &var)
    return ave*volume,np.sqrt(var)*volume

