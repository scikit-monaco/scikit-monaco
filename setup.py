
import os
import sys

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
    metadata = dict(**extra_setuptools_args)

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

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path="").todict())
