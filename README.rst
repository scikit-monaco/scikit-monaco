
scikit-monaco
=============

scikit-monaco is a library for Monte Carlo integration in python. The core is
written in Cython, with process-level parallelism to squeeze the last bits of
speed out of the python interpreter.

A code snippet is worth a thousand words. Let's look at integrating 
``sqrt(x**2 + y**2 + z**2)`` in the unit square::

    >>> from skmonaco import mcquad
    >>> from math import sqrt
    >>> result, error = mcquad(lambda (x,y,z): sqrt(x**2+y**2+z**2), 
    ...     npoints=1e6, xl=[0.,0.,0.], xu=[1.,1.,1.])
    >>> print "{} +/- {}".format(result,error)
    0.960695982212 +/- 0.000277843266684

Links
-----

* Home page: 
* Source code: https://github.com/scikit-monaco/scikit-monaco
* Issues: https://github.com/scikit-monaco/scikit-monaco/issues

Installation
------------

From Pypi
^^^^^^^^^

From source
^^^^^^^^^^^

Clone the repository using::
    
    $ git clone https://github.com/scikit-monaco/scikit-monaco.git

And run::

    $ python setup.py install

in the project's root directory.


Testing
-------

After the installation, run ``$ python runtests.py`` in the package's root directory.


Issue reporting and contributing
--------------------------------

Report issues using the `github issue tracker <https://github.com/scikit-monaco/scikit-monaco/issues>`_.

Read the CONTRIBUTING guide to learn how to contribute.
