
from __future__ import print_function

import vegas
from collections import namedtuple

npoints_tuple = namedtuple("npoints_tuple", ["niterations", "nevaluations"])

NITERATIONS_DEFAULT = 10

def mcvegas(f, npoints, xl, xu):
    """
    Compute a definite integral using the VEGAS algorithm.

    This function integrates `f` using the VEGAS algorithm. It is a wrapper
    around P. Lepage's vegas package.

    Parameters
    ----------
    f: function
        The integrand. Must take an iterable of length `d`, where `d`
        is the dimensionality of the integral, as argument, and return
        a float or an iterable of floats.
    npoints: int or (int, int) pair
        An upper bound on the number of points. If passed as an integer,
        the points will be divided evenly over ten iterations. If passed
        as a tuple, the first value is the number of iterations and the 
        second is the number of evalutations per iteration.
    xl, xu : iterable
        Iterable of length `d`, where `d` is the dimensionality of the 
        integrand. `xl` denotes the bottom left corner and `xu` the top
        right corner of the integration region.
    """
    try:
        niterations, nevaluations = npoints
    except TypeError:
        niterations = NITERATIONS_DEFAULT
        nevaluations = npoints/niterations
    integrator = vegas.Integrator(zip(xl, xu))
    result = integrator(f, nitn=niterations, neval=nevaluations)
    return result.mean, result.sdev
