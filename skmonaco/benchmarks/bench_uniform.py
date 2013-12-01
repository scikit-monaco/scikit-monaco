
from __future__ import print_function
from numpy.testing import TestCase,measure,run_module_suite

from skmonaco import mcquad

class TestUniform(TestCase):

    def setUp(self):
        self.constant = lambda x: 1.0

    def run_print(self,test_list):
        print()
        print(" Integrating constant function")
        print(" =============================")
        print()
        print(" ndims | npoints | nprocs | time ")
        print(" ------------------------------- ")
        
        for ndims,npoints,nprocs,repeat in test_list:
            print(" {ndims:5} | {npoints:7} | {nprocs:6} |".format(ndims=ndims,npoints=npoints,nprocs=nprocs),end="")
            xl = [0.]*ndims
            xu = [1.]*ndims
            time = measure("mcquad(lambda x: 1.0,{npoints},{xl},{xu},nprocs={nprocs})".format(npoints=npoints,xl=str(xl),xu=str(xu),nprocs=str(nprocs)),repeat)
            print(" {time:.2f}  (seconds for {ncalls} calls)".format(time=time,ncalls=repeat))


    def bench_constant_serial(self):
        test_list = [ (1,12500,1,50),(1,25000,1,50),(1,50000,1,50), 
                (1,100000,1,50), (2,50000,1,50), (3,50000,1,50),
                (4,50000,1,50),(5,50000,1,50),(10,50000,1,50)]
        self.run_print(test_list)

    def bench_constant_parallel(self):
        test_list = [ (3,1000000,1,5),(3,1000000,2,5), (3,1000000,4,5)]
        self.run_print(test_list)
        print()
        print("  Warning: Timing for nprocs > 1 are not correct.")
        print()

if __name__ == '__main__':
    run_module_suite()
