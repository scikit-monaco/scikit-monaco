
"""
=======================
Monte Carlo integration
=======================

This module provides a toolkit for Monte Carlo integration.

    mcquad   -- Integration over a uniformly-sampled hypercube.
    mcimport -- Integration over points distributed according to a particular pdf.
    integrate_from_points -- Integration of a function over specific points.
"""

__version__ = "0.1-git"

from uniform import * 
from importance import *
from from_pts import *

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
