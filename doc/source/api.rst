
Reference Guide
===============

.. currentmodule:: skmonaco

Uniform sampling Monte Carlo integration
----------------------------------------

`mcquad` samples uniformly from a hypercube. This method can also be used to
integrate over more complicated volumes using the procedure described in
:ref:`complex-integration-volumes`. It will lead to large errors if the
integration region is large and the integrand changes rapidly over a small
fraction of the total integration region.

.. autofunction:: mcquad

Importance sampling
-------------------

In importance sampling, the integrand is factored
into the product of a probability density :math:`\rho(x)` and another function
:math:`h(x)`:

.. math::

    f(x) = \rho(x) h(x)

The integration proceeds by sampling from :math:`\rho(x)` and calculating
:math:`h(x)` at each point. In `scikit-monaco`, this is achieved with the
`mcimport` function.

.. autofunction:: mcimport 

MISER Monte Carlo
-----------------

`mcmiser` samples from a hypercube using the MISER algorithm, and
can also be used to integrate over more complicated volumes using the procedure
described in :ref:`complex-integration-volumes`. The algorithm is adaptive,
inasmuch as it will use more points in regions where the variance of the
integrand is large. It is almost certainly likely to be superior to `mcquad`
for "complicated" integrands (integrands which are smooth over a large fraction
of the integration region but with large variance in a small region), with
dimensionality below about 6.

.. autofunction:: mcmiser

VEGAS Monte Carlo
-----------------

`mcvegas` samples from a hypercube using the VEGAS algorithm. As with `mcmiser`,
more complex integration volumes can be achieved as described in 
:ref:`complex-integration-volumes`. 
The algorithm is adaptive, so will use more points in
regions where the variance of the integral is large.

`mcvegas` relies on the Python package `Vegas
<https://pypi.python.org/pypi/vegas>`__, produced by Peter Lepage, who first
proposed the VEGAS algorithm. Note that the Vegas package is licensed under the
GPL.

.. autofunction:: mcvegas

Utility functions
-----------------

.. autofunction:: integrate_from_points
