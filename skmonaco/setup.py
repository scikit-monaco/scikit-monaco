
def configuration(parent_package="",top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    config = Configuration("skmonaco",parent_package,top_path)

    config.add_extension("_mc",sources="_mc.c",include_dirs=[numpy.get_include()],
            libraries=["m"])
    
    config.add_data_dir("tests")
    config.add_data_dir("benchmarks")
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path="").todict())
