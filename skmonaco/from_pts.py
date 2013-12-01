
from __future__ import division, absolute_import

from .mc_base import _MC_Base
from . import _mc

__all__ = [ "integrate_from_points" ]

class _Integrator_From_Points(_MC_Base):

    def __init__(self,f,points,args=(),nprocs=1,batch_size=None,
            weight=1.0):
        self.f = f
        self.points = points
        self.args = args
        self.weight = weight
        _MC_Base.__init__(self,nprocs,batch_size)

    @property
    def npoints(self):
        return len(self.points)

    def create_batches(self):
        _MC_Base.create_batches(self)
        self.batch_start = []
        self.batch_end = []
        accum = 0
        for batch_size in self.batch_sizes:
            self.batch_start.append(accum)
            self.batch_end.append(accum+batch_size)
            accum += batch_size

    def make_integrator(self):
        f = self.f
        def func(batch_number):
            start = self.batch_start[batch_number]
            end = self.batch_end[batch_number]
            return _mc.integrate_points(f,self.points[start:end],self.weight,self.args)
        return func


def integrate_from_points(f,points,args=(),nprocs=1,batch_size=None,weight=1.0):
    """
    Compute a definite integral over a set of points.

    This routine evaluates `f` for each of the points passed, 
    returning the average value and variance.

    Parameters
    ----------
    f : function
        A Python function or method to integrate. It must take an iterable
        of length `d`, where `d` is the dimensionality of the integral,
        as argument, and return either a float or a numpy array.
    points : numpy array
        A numpy array of shape ``(npoints,dim)``, where `npoints` is
        the number of points and `dim` is the dimentionality of 
        the integral.

    Other Parameters
    ----------------
    nprocs : int >= 1, optional
        Number of processes to use concurrently for the integration. Use 
        nprocs=1 to force a serial evaluation of the integral. This defaults
        to the value returned by multiprocessing.cpu_count().
    batch_size : int, optional
        The integration is batched, meaning that `batch_size` points are 
        generated, the integration is run with these points, and the 
        results are stored. Each batch is run by a single process. It may 
        be useful to reduce `batch_size` if the dimensionality of the 
        integration is very large.

    Returns
    -------
    value : float or numpy array.
        The estimate for the integral. If the integrand returns an array,
        this will be an array of the same shape.
    error : float or numpy array
        An estimate for the error (the integral has, approximately, a 0.68 
        probability of being within `error` of the correct answer).
 
    Examples
    --------

    Integrate x*y over the unit square.

    >>> from numpy.random import ranf
    >>> npoints = int(1e5)
    >>> points = ranf(2*npoints).reshape((npoints,2)) # Generate some points
    >>> points.shape
    (100000,2)
    >>> integrate_from_points(lambda x_y:x_y[0]*x_y[1], points) 
    (0.24885..., 0.00069...)
    """
    return _Integrator_From_Points(
            f,points,args,nprocs,batch_size,weight).run()

