
Reference Guide
===============

.. currentmodule:: skmonaco

Uniform sampling Monte-Carlo integration
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

Utility functions
-----------------

.. autofunction:: integrate_from_points
