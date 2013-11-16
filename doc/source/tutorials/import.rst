
Importance sampling
===================

.. currentmodule:: skmcquad

The techniques described in the previous sections work well if the 
variance of the integrand is similar over the whole integration volume. As a counter-example, consider the integral :math:`\int_{-\infty}^\infty \cos(x)^2 e
^{-x^2/2}\,dx`. Only the area around :math:`x = 0` contributes to the integral in a significant way, while the tails, which stretch out to infinity, contribute almost nothing. 

We would therefore like a method that samples more thoroughly regions where the integral varies rapidly. This is the idea behind **importance sampling**. 

To use importance sampling, we need to factor the integrand :math:`f(x)` into a probability distribution :math:`\rho(x)` and another function :math:`h(x)`. For the integrand described above, we can take :math:`\rho(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}` (normal distribution) and :math:`h(x) = \sqrt{2\pi}\cos^2(x)`. We can then draw samples from :math:`g(x)` and, for each sample, calculate :math:`h(x)`. This is done using the function :func:`mcimport`:

.. code:: python
    
    >>> from skmcquad import mcimport
    >>> from numpy.random import normal
    >>> from math import cos,sqrt,pi
    >>> result, error = mcimport(lambda x: sqrt(2.*pi)*cos(x)**2, # h(x)
    ...    npoints = 100000,
    ...    distribution = normal # rho(x)
    ...    )
    >>> result # Correct answer: 1.42293
    1.423388348518721
    >>> error
    0.002748743084305

We see that :func:`mcimport` takes at least three arguments, the first one being the function :math:`h(x)`, that is, the integrand divided by the probability distribution from which we sample, the second being the number of points and the third is a function that returns random samples from the probability distribution :math:`\rho(x)`.  The ``numpy.random`` module provides many useful distributions that can be passed to ``mcimport``. :func:`mcimport` returns two numbers: the first corresponds to the integral estimate, and the second corresponds to the estimated error. 

Multi-dimensional integrals
---------------------------

To look at a slightly more complicated example, let's integrate :math:`f(x,y) = e^{-(y+2)}` in the truncated ring described :ref:`above <ring-figure>`. The integrand is largest around :math:`y = -2`, and decays very quickly:

.. figure:: import_integrand.*
    :height: 350pt
    :align: center

    :math:`e^{-(y+2)}` in the truncated ring.

It makes sense to try and concentrate sample points around :math:`y = -2`. We can do this by sampling from the exponential distribution (centered about :math:`y=-2`) along `y` and uniformly along `x` (in the region :math:`1 \le x < 3`), such that :math:`\rho(x,y) = \frac{1}{2}e^{-(y+2)}`, where the pre-factor of :math:`1/2` arises from the normalisation condition on the uniform distribution along `x`). The `distribution` function that must be passed to :func:`mcimport` must take a `size` argument and return an array of shape ``(size,d)`` where `d` is the number of dimensions of the integral. This could be achieved with the following function:

.. code:: python
    
    from numpy.random import exponential, uniform

    def distribution(size):
        """
        Return `size` (x,y) points distributed according to a uniform distribution
        along x and an exponential distribution along y.
        """
        xs = uniform(size=size,low=1.0,high=3.0)
        ys = exponential(size=size,scale=1.0) - 2.
        return np.array((xs,ys)).T

    distribution(100).shape
    # (100,2)

This is what 200 points distributed according to ``distribution`` look like. Points that fall within :math:`\Omega` are colored in red, and those that fall outside are colored in black. Evidently, points are concentrated about :math:`y = -2`, such that this region gets sampled more often. 

.. figure:: import_plot.*
    :height: 350pt
    :align: center

The function to sample is :math:`f(x,y) = 2` (the factor of 2 cancels the ``1/(high-low)`` prefactor in the uniform distribution) if :math:`f(x,y) \in \Omega` and 0 otherwise.

.. code:: python

    def f((x,y)):
        """
        The integrand.
        """
        r = np.sqrt(x**2 + y**2)
        if r >= 2. and r <= 3. and x >= 1. and y >= -2.:
            # (x,y) in correct volume
            return 2.
        else:
            return 0.

    mcimport(f,1e5,distribution,nprocs=4)
    # (1.18178, 0.0031094848254976256)

Choosing a probability distribution
-----------------------------------

Generally, it is far from obvious how to split the integrand :math:`f(x)` into a probability distribution :math:`\rho` and an integration kernel :math:`h(x)`. It can be shown [NR]_ that, optimally :math:`\rho \propto |f|`. Unfortunately, this is likely to be very difficult to sample from. 

The following recipe is relatively effective:

1. If the integrand decays to 0 at the edge of the integration region, find a distribution that has the same rate of decay, or decays slightly slower. Thus, if, for instance, the integrand if :math:`p(x) e^{-x^2}`, where :math:`p(x)` is a polynomial, sample from a normal distribution with :math:`\sigma=1/\sqrt{2}`, or slightly less if :math:`p(x)` contains high powers that delay the onset of the exponential decay.

2. Locate the mode of your distribution at the same place as the mode of the integrand.

The following two cases should be avoided:

1. The probability distribution should not be low in places where the integrand is large.

2. Somewhat less importantly, the probability distribution should not be high in regions where the variance of the integrand is low. 


.. [NR] Press, W. H, Teutolsky, S. A., Vetterling, W. T., Flannery, B. P., Numerical Recipes, The art of scientific computing, 3rd edition, Cambridge University Press (2007).
