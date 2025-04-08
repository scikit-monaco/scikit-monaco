.. Automatically generated from LONG_DESCRIPTION keyword in 
.. setup.py. Do not edit directly.

scikit-monaco
=============

scikit-monaco is a library for Monte Carlo integration in `Python <https://www.python.org/>`_.
The core is written in `Cython <http://cython.org/>`_, with process-level parallelism
to squeeze the last bits of speed out of the python interpreter.

A code snippet is worth a thousand words. Let's look at integrating
``sqrt(x**2 + y**2 + z**2)`` in the unit square:

.. code:: python

    >>> from skmonaco import mcquad
    >>> from math import sqrt
    >>> result, error = mcquad(
    ...     lambda xs: sqrt(xs[0]**2 + xs[1]**2 + xs[2]**2),
    ...     npoints=1e6, xl=[0.,0.,0.], xu=[1.,1.,1.])
    >>> print("{} +/- {}".format(result,error))
    0.960695982212 +/- 0.000277843266684

Links
-----

* `Home page <https://pypi.python.org/pypi/scikit-monaco>`_
* `Documentation <http://scikit-monaco.readthedocs.org/en/latest/>`_
* `Source code <https://github.com/scikit-monaco/scikit-monaco>`_
* `Issues <https://github.com/scikit-monaco/scikit-monaco/issues>`_

Installation
------------

**The installation from Pypi seems to have been broken for quite some time**
(see <https://github.com/scikit-monaco/scikit-monaco/issues/14> or
<https://github.com/scikit-monaco/scikit-monaco/issues/16>), **and we are working on it.**

**As of April 8th 2025, a partial fix is to install @Naereen's development version,
directly from GitHub**::

    $ pip install --upgrade git+https://github.com/Naereen/scikit-monaco

From Pypi
^^^^^^^^^

The easiest way to download and install scikit-monaco is from the Python
package index (pypi). Just run::

    $ python easy_install scikit-monaco

Or, if you have pip::

    $ pip install scikit-monaco

From source
^^^^^^^^^^^

Clone the repository using::

    $ git clone https://github.com/scikit-monaco/scikit-monaco.git

And run::

    $ # python setup.py install # this is deprecated
    $ pip install .

in the project's root directory.


Testing
-------

The testing (and benchmarking) is broken as well since a few years
(see <https://github.com/scikit-monaco/scikit-monaco/issues/15>
and <https://github.com/scikit-monaco/scikit-monaco/issues/17>),
we will work on fixing it soon.

After the installation, run ``$ python runtests.py`` in the package's root directory.


Issue reporting and contributing
--------------------------------

Report issues using the `github issue tracker <https://github.com/scikit-monaco/scikit-monaco/issues>`_.

Read the `CONTRIBUTING <https://github.com/scikit-monaco/scikit-monaco/blob/master/CONTRIBUTING.rst>`_ guide to learn how to contribute.
