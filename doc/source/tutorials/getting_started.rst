
Getting started
===============

.. currentmodule:: skmonaco

**Scikit-monaco** is a toolkit for Monte Carlo integration. It is
written in Cython for efficiency and includes parallelism to take
advantage of processors with multiple core.

Let's suppose that we want to calculate this integral: :math:`\int_0^1 \int_0^1 x \cdot y \, dx dy`. 
This could be computed using :func:`mcquad`.

.. code:: python

    >>> from skmonaco import mcquad
    >>> mcquad(lambda (x,y): x*y, # integrand
    ...     xl=[0.,0.],xu=[1.,1.], # lower and upper limits of integration
    ...     npoints=100000 # number of points
    ...     ) 
    (0.24959359250821114, 0.0006965923631156234)

:func:`mcquad` returns two numbers. The first (0.2496...) is the value of
the integral. The second is the estimate in the error (corresponding,
roughly, to one standard deviation). Note that the correct answer in
this case is :math:`1/4`.

The :func:`mcquad` call distributes points uniformly in a hypercube and sums 
the value of the integrand over those points. 
The first argument to :func:`mcquad` is a callable specifying the integrand.
This can often be done conveniently using a lambda function. The integrand must
take a single argument: a numpy array of length `d`, where `d` is the number of 
dimensions in the integral.  For instance, the following would be valid integrands:

.. code:: python
    
    >>> from math import sin,cos
    >>> integrand = lambda x: x**2
    >>> integrand = lambda (x,y): sin(x)*cos(y)
    >>> integrand = lambda xs: sum(xs)

If the integrand takes additional parameters, they can be passed to the
function through the `args` argument. Suppose that we want to evaluate the
product of two Gaussians with exponential factors ``alpha`` and ``beta``:

.. code:: python
    
    >>> import numpy as np
    >>> from skmonaco import mcquad
    >>> f = lambda (x,y),alpha,beta: np.exp(-alpha*x**2)*np.exp(-beta*y**2)
    >>> alpha = 1.0
    >>> beta = 2.0
    >>> mcquad(f,xl=[0.,0.],xu=[1.,1.],npoints=100000,args=(alpha,beta))
    (0.44650031245386379, 0.00079929285076240579)

:func:`mcquad` runs on a single core as default. The parallel behaviour can be
controlled by the `nprocs` parameter. The following can be run in an
ipython session to take advantage of their timing routines.

.. code:: python

    >>> from math import sin,cos
    >>> from skmonaco import mcquad
    >>> f = lambda x,y: sin(x)*cos(y)
    >>> %timeit mcquad(f,xl=[0.,0.],xu=[1.,1.],npoints=1e6,nprocs=1) 
    1 loops, best of 3: 2.26 s per loop
    >>> %timeit mcquad(f,xl=[0.,0.],xu=[1.,1.],npoints=1e6,nprocs=2) 
    1 loops, best of 3: 1.36 s per loop
    

.. _complex-integration-volumes:

Complex integration volumes
===========================

Monte Carlo integration is very useful when calculating integrals over
complicated integration volumes. Consider the integration volume shown in red 
in this figure:

.. _ring-figure:

.. figure:: build/rings.*
    :height: 350pt
    :align: center

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

.. image:: build/scatter.*
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

    mcquad(g,npoints=100000,xl=[1.,-2.],xu=[3.,3.],nprocs=4)
    # (9.9161745618624231, 0.049412524880183335)


Array-like integrands
=====================

We are not limited to integrands that return floats. We can have integrands
that return array objects. This can be useful for calculating several integrals
at the same time. For instance, let's say that we want to calculate
both :math:`x^2` and :math:`y^2` in the volume :math:`\Omega` described in the
previous section. We can do both these integrals simultaneously.

.. code:: python

    import numpy as np
    
    def f((x,y)):
        """ f((x,y,)) now returns an array with both integrands. """
        return np.array((x**2,y**2))
    
    def g((x,y)):
        r = np.sqrt(x**2 + y**2)
        if r >= 2. and r <= 3. and x >= 1. and y >= -2.:
            # (x,y) in correct volume
            return f((x,y))
        else:
            return np.zeros((2,))

    result, error = mcquad(g,npoints=100000,xl=[1.,-2.],xu=[3.,3.],nprocs=4)
    print result
    # [ 23.27740875   9.89103493] 
    print error
    # [ 0.08437993  0.04938343]

We see that if the integrand returns an array, :func:`mcquad` will return two
arrays of the same shape, the first corresponding to the values of the
integrand and the second to the error in those values.
