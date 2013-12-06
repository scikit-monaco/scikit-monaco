
import os
import sys

DISTNAME = "scikit-monaco"
DESCRIPTION = "Python modules for Monte Carlo integration"
MAINTAINER = "Pascal Bugnion"
MAINTAINER_EMAIL = "pascal@bugnion.org"
URL = "https://pypi.python.org/pypi/scikit-monaco"
LICENSE = "new BSD"

LONG_DESCRIPTION = """

scikit-monaco
=============

scikit-monaco is a library for Monte Carlo integration in python. The core is
written in Cython, with process-level parallelism to squeeze the last bits of
speed out of the python interpreter.

A code snippet is worth a thousand words. Let's look at integrating 
``sqrt(x**2 + y**2 + z**2)`` in the unit square:

.. code:: python

    >>> from skmonaco import mcquad
    >>> from math import sqrt
    >>> result, error = mcquad(
    ...     lambda xs: sqrt(xs[0]**2+xs[1]**2+xs[2]**2), 
    ...     npoints=1e6, xl=[0.,0.,0.], xu=[1.,1.,1.])
    >>> print "{} +/- {}".format(result,error)
    0.960695982212 +/- 0.000277843266684

Links
-----

* Home page: https://pypi.python.org/pypi/scikit-monaco
* Documentation: http://scikit-monaco.readthedocs.org/en/latest/
* Source code: https://github.com/scikit-monaco/scikit-monaco
* Issues: https://github.com/scikit-monaco/scikit-monaco/issues

Installation
------------

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

    $ python setup.py install

in the project's root directory.


Testing
-------

After the installation, run ``$ python runtests.py`` in the package's root directory.


Issue reporting and contributing
--------------------------------

Report issues using the `github issue tracker <https://github.com/scikit-monaco/scikit-monaco/issues>`_.

Read the CONTRIBUTING guide to learn how to contribute.
"""

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

# This is a bit hackish: we are setting a global variable so that the main
# skmonaco __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet. While ugly, it's
# a lot more robust than what was previously being used.
# Copied from scipy setup file.
builtins.__SKMONACO_SETUP__ = True

def write_readme():
    """
    Create README file from LONG_DESCRIPTION, replacing non-standard
    bits of re-structured text.
    """
    with open("README.rst","w") as f:
        f.write("""\
.. Automatically generated from LONG_DESCRIPTION keyword in 
.. setup.py. Do not edit directly.\
""")
        f.write(LONG_DESCRIPTION.replace(".. code:: python","::"))

import skmonaco

VERSION = skmonaco.__version__


# For some commands, use setuptools.
if len(set(('develop', 'release', 'bdist_egg', 'bdist_rpm',
           'bdist_wininst', 'install_egg_info', 'build_sphinx',
           'egg_info', 'easy_install', 'upload',
           '--single-version-externally-managed',
            )).intersection(sys.argv)) > 0:
    import setuptools
    extra_setuptools_args = dict(
        zip_safe=False, # the package can run out of an .egg file
        include_package_data=True,
    )
else:
    extra_setuptools_args = dict()

def configuration(parent_package="",top_path=None):
    if os.path.exists("MANIFEST"):
        os.remove("MANIFEST")
    write_readme()
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None,parent_package,top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('skmonaco')
    
    return config

def setup_package():
    metadata = dict(
            name=DISTNAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            license=LICENSE,
            url=URL,
            version=VERSION,
            long_description=LONG_DESCRIPTION,
            classifiers=[
                'Intended Audience :: Science/Research',
                'Intended Audience :: Developers',
                'Intended Audience :: Financial and Insurance Industry',
                'License :: OSI Approved :: BSD License',
                'Programming Language :: Cython',
                'Programming Language :: Python',
                'Topic :: Software Development',
                'Topic :: Scientific/Engineering',
                'Operating System :: POSIX',
                'Operating System :: Unix',
                'Operating System :: MacOS',
                'Programming Language :: Python :: 2',
                'Programming Language :: Python :: 2.7',
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.3'],
            **extra_setuptools_args)

    if (len(sys.argv) >= 2 and
            ("--help" in sys.argv[1:] or
             sys.argv[1] in ("--help-commands","egg_info","--version","clean"))):
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

    else:
        from numpy.distutils.core import setup
        metadata["configuration"] = configuration
    setup(**metadata)

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup_package()
