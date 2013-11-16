
Reference Guide
===============

Uniform sampling Monte-Carlo integration
----------------------------------------

`mcquad` samples uniformly from a hypercube. This method can also be used to
integrate over more complicated volumes using the procedure described in
:ref:`complex-integration-volumes`. It will lead to large errors if the
integration region is large and the integrand changes rapidly over a small
fraction of the total integration region.

.. autofunction:: skmcquad.mcquad
