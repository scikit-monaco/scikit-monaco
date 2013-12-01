
Installation
============

Dependencies
------------

scikit-monaco requires the following:

* Python (tested on 2.7 and 3.3)
* NumPy (tested on 1.8)

scikit-monaco may work with other versions of python and numpy, but these
are currently not supported.

You will also need the python development headers and a working C compiler. On
Debian-based operating systems such as Ubuntu, you can install the requirements
with::

    sudo apt-get install python-dev python-numpy

Additionally, you will need the `nose` package to run the test suite. This can
be installed with::
    
    sudo apt-get install python-nose

Installing using easy_install or pip
-------------------------------------

The easiest way to install a stable release is using `pip` or `easy_install`.
Run either::

    pip install -U scikit-monaco

or::

    easy_install -U scikit-monaco
    
This will automatically fetch the package from Pypi and install it. If your
python installation is system-wide, you will need to run these as root.

Installing from source
----------------------

Download the source code from the `Python package index
<https://pypi.python.org/pypi/scikit-monaco>`_, extract it, move into the 
root directory of the project and run the installer::

    python setup.py install

Installing the development version
----------------------------------

scikit-monaco is version controlled under `git <http://git-scm.com/>`_. The
repository is hosted on `github
<https://github.com/scikit-monaco/scikit-monaco>`_. Clone the repository
using::

    git clone https://github.com/scikit-monaco/scikit-monaco.git

Move to the root of the project's directory and run::

    python setup.py install

To build the documentation, run ``python setup.py develop`` to build
scikit-monaco in the source directory, then ``cd doc; make html`` to build the
html version of the documentation.

Testing
-------

Running ``python runtests.py`` in the project's root directory will run the
test suite.

Benchmarks
----------

There are some benchmarks available. These can be run using::

    python -c "import skmonaco; skmonaco.bench()"

