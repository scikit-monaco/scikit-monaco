
import numpy as np
from numpy.testing import build_err_msg
from numpy.testing.nosetester import import_nose
import sys

def within_tol(a,b,tol):
    return np.abs(a-b).max() < tol

def assert_within_tol(a,b,tol,err_msg=""):
    if not within_tol(a,b,tol):
        msg = build_err_msg((a,b),err_msg)
        raise AssertionError(msg)

def run_module_suite(file_to_run=None,argv=None):
    """
    Overload numpy's run_module_suite to allow passing arguments to nose.

    Any argument passed to 'argv' will be passed onto nose.

    For example, 'run_module_suite(argv=["","-A","not slow"])' will only run
    tests that are not labelled with the decorator 'slow'.

    This is particularly useful if argv is passed 'sys.argv'.
    """
    if file_to_run is None:
        f = sys._getframe(1)
        file_to_run = f.f_locals.get('__file__', None)
        if file_to_run is None:
            raise AssertionError
    if argv is None:
        argv = [""]
    import_nose().run(argv=argv+[file_to_run])

