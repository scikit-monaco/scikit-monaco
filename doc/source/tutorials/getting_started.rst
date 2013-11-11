
Getting started
===============

**Scikit-mcquad** is a toolkit for Monte Carlo integration. It is
written in Cython for efficiency and includes parallelism to take
advantage of processors with multiple core.

Let's suppose that we want to calculate this integral: :math:`\int_0^1 \int_0^1 x \cdot y \, dx dy`:

.. code:: python

    >>> from skmcquad import mcquad
    >>> mcquad(lambda (x,y): x*y, # integrand
    ...     xl=[0.,0.],xu=[1.,1.], # lower and upper limits of integration
    ...     npoints=100000 # number of points
    ...     ) 

    (0.24959359250821114, 0.0006965923631156234)

``mcquad`` returns two numbers. The first (0.2496...) is the value of
the integral. The second is the estimate in the error (corresponding,
roughly, to one standard deviation). Note that the correct answer in
this case is :math:`1/4`.

The ``mcquad`` call distributes points uniformly in a hypercube and sums 
the value of the integrand over those points. The integrand is specified 
as a function ``lambda xs : f(xs)`` whose first argument must be a list
of length `d` describing a point in the integration volume (where `d` is 
the number of dimensions of the problem).

Complex integration volumes
---------------------------

Monte Carlo integration is very useful when calculating integrals over
complicated integration volumes. Consider the integration volume shown in red 
in this figure:

.. image:: rings.*
    :height: 350pt

The integration volume is the intersection of a ring bounded by circles 
of radius 2 and 3, and a large square.

The volume of integration :math:`\Omega` is given by the following three
inequalities:

-  :math:`\sqrt{x^2 + y^2} \ge 2` : inner circle
-  :math:`\sqrt{x^2 + y^2} \le 3` : outer circle
-  :math:`x \ge 1` : left border of box
-  :math:`y \ge -2` : top border of box

Let :math:`\Omega` denote this volume. Suppose that we want to find the integral
of :math:`f(x,y)` over this region. We can re-cast the integral over
:math:`\Omega` as an integral over a square that completely encompasses
:math:`\Omega`. In this case, the smallest rectangle to contain :math:`\Omega` starts
has bottom left corner :math:`(1,-2)` and top right corner :math:`(3,3)`. This
rectangle is shown in blue in the figure below.

Then:

:math:`\iint_\Omega f(x,y) \, dx dy = \int_{-2}^3 \int_1^3 g(x,y) \,dx dy`

where

:math:`g(x,y) = \left\{ \begin{array}{l l}f(x,y) & (x,y) \in \Omega \\ 0 & \mathrm{otherwise}\end{array}\right.`

This is illustrated in the figure below, which shows 200 points randomly
distributed within the rectangle bounded by the blue lines. Points in red fall
within :math:`\Omega` and thus contribute to the integral, while points in
black fall outside :math:`\Omega`.

.. image:: scatter.*
    :height: 350pt

To give a concrete example, let's take :math:`f(x,y) = y^2`.

.. code:: python

    import numpy as np
    
    def f((x,y)):
        return y**2
    
    def g((x,y)):
        """
        The integrand.
        """
        r = np.sqrt(x**2 + y**2)
        if r >= 2. and r <= 3. and x >= 1. and y >= -2.:
            # (x,y) in correct volume
            return f((x,y))
        else:
            return 0.

To speed up the calculation, we can run the integration in parallel.
This is done by passing an ``nprocs`` keyword argument (note that this
only results in a speedup on a multi-core processor):

.. code:: python

    >>> from skmcquad import mcquad
    >>> mcquad(g,npoints=100000,xl=[1.,-2.],xu=[3.,3.],nprocs=4)
    (9.9161745618624231, 0.049412524880183335)


