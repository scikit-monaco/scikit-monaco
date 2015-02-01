
from __future__ import print_function

import vegas
from collections import namedtuple

Npoints = namedtuple("Npoints", ["niterations", "nevaluations"])

NITERATIONS_DEFAULT = 10

def mcvegas(f, npoints, xl, xu, full_output=False):
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
    npoints: int, (int, int) pair or Npoints object
        An upper bound on the number of points. If passed as an integer,
        the points will be divided evenly over ten iterations. If passed
        as a tuple, the first value is the number of iterations and the 
        second is the number of evalutations per iteration.
    xl, xu : iterable
        Iterable of length `d`, where `d` is the dimensionality of the 
        integrand. `xl` denotes the bottom left corner and `xu` the top
        right corner of the integration region.
    full_output : boolean
        When true, the return value contains, as its third element, a 
        dictionary with additional information. See `full_output` below.

    Returns
    -------
    value : float
        The estimate for the integral.

    error : float
        An estimate for the error, corresponding to one standard 
        deviation. The integral has, approximately, a 0.68 
        probability of being within `error` of the correct answer.

    additional_output : dictionary
        If `full_output` is set to True, `vegas` returns a dictionary with
        additional information. This dictionary contains the following values:

        ``chi2`` : The average squared deviation of errors for each , normalised by
                 the standard deviation for that iteration. If this number is 
                 significantly larger than the number of iterations (plus or 
                 minus its own square root), the distribution of errors is non-
                 gaussian, so the value returned by `error` is likely to 
                 underestimate the true error in the integral.

        ``Q`` : p-value for chi2. If this number is less than 0.1, `error` is
              likely to underestimate the true error in the integral. 

        ``itn_results`` : the results at each iteration of the Vegas algorithm.
        
        ``itn_summary`` : a tabular representation of the results at each
                        iteration.

    Notes
    -----
    This function is a thin wrapper around P. Lepage's vegas module [vegas]_, 
    which is under the GPL. 

    References
    ----------
    .. [vegas] G. P. Lepage 
            https://pypi.python.org/pypi/vegas

    Examples
    --------
    Integrate x*y over the unit square. The correct value is 1/4.

    >>> mcvegas(lambda x: x[0]*x[1], npoints=20000, xl=[0.,0.], xu=[1.,1.])
    (0.249883... 0.000116...)

    Note that this is about fifteen times more accurate than the equivalent
    call to `mcquad`, for the same number of points.
    """
    try:
        niterations, nevaluations = npoints
    except TypeError:
        niterations = NITERATIONS_DEFAULT
        nevaluations = npoints/niterations
    integrator = vegas.Integrator(zip(xl, xu))
    result = integrator(f, nitn=niterations, neval=nevaluations)
    if full_output:
        additional_output = dict(
            chi2=result.chi2, Q=result.Q, 
            itn_results=result.itn_results, itn_summary=result.summary()
        )
        return result.mean, result.sdev, additional_output
    else: 
        return result.mean, result.sdev
