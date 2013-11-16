
"""
=======================
Monte Carlo integration
=======================

This module provides a toolkit for Monte Carlo integration.

    mcquad   -- Integration over a uniformly-sampled hypercube.
"""

from uniform import * 
from importance import *

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
