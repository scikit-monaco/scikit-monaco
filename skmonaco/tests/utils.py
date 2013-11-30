
import numpy as np
from numpy.testing import build_err_msg

def within_tol(a,b,tol):
    return np.abs(a-b).max() < tol

def assert_within_tol(a,b,tol,err_msg=""):
    if not within_tol(a,b,tol):
        msg = build_err_msg((a,b),err_msg)
        raise AssertionError(msg)
