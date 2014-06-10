
import numpy as np
from numpy.random import randint

from skmonaco import mcmiser

miser_functions = { 
        "x**2" : lambda x: x**2,
        "x**2 (sum)" : lambda x: sum(x**2),
        "x**2+3*y**2" : lambda x: x[0]**2 + 3*x[1]**2,
        "prod" : lambda x: np.prod(x)
}

class TestRun(object):

    def __init__(self,fcode,npoints,ntrials,lbound, ubound, nprocs, analytical_value):
        """
        Create a TestFixture.

        Arguments
        ---------

        fcode : str
            name of function in `miser_functions` dictionary.

        npoints : int
            number of MC integration points in each trial.

        ntrials : int
            number of trials to get approximate values of the mean
            error and standard deviation in error.

        analytical_value : float
            analytical value of the integral.

        lbound, ubound : list of floats
            lower (upper) bound for the integration

        nprocs : int
            number of processes to use in the integration.

        Example
        -------

        To generate a test fixture, 

        >>> fixture = TestRun("x**2", 1e4, 100, [0.], [1.], 1, 1./3.)
        >>> fixture.check_results_error()
        >>> fixture.mean_sigma # average error per trial
        3.86390778798e-05
        >>> fixture.sigma_sigma  # error in error per trial
        3.45037432946e-07
        >>> with open("fixture.pkl") as f:
                pickle.dump(fixture,f)

        Then, to run a regression test,

        >>> import miser_data_generator
        >>> with open("fixture.pkl") as f:
                fixture = pickle.load(f)
        >>> fixture.check_trial_run()
        """
        self.fcode = fcode
        self.f = miser_functions[self.fcode]
        self.npoints = npoints
        self.ntrials = ntrials
        self.analytical_value = analytical_value
        self.lbound = lbound
        self.ubound = ubound
        self.nprocs = nprocs
        self.mean_sigma = None # Calculated by self.check_results_error()
        self.sigma_sigma = None # Calculated by self.check_results_error()
        self.seed = randint(0,10000)

    def __getstate__(self):
        result = self.__dict__.copy()
        del result["f"]
        return result

    def __setstate__(self,pickled_dict):
        self.__dict__ = pickled_dict
        self.f = miser_functions[self.fcode]

    def _get_results_errors(self):
        results, errors = [], []
        for itrial in range(self.ntrials):
            res, err = mcmiser(self.f,self.npoints,self.lbound,self.ubound,
                    seed=itrial+self.seed)
            results.append(res)
            errors.append(err)
        results = np.array(results)
        errors = np.array(errors)
        return results, errors

    def check_results_errors(self, verbose=False):
        results, errors = self._get_results_errors()
        mean_result = results.mean()
        mean_error = errors.mean()
        error_in_mean = abs(mean_result-self.analytical_value)
        if verbose:
            print ("  <sigma> = {}; expect < {} (approximately).".format(
                    error_in_mean,mean_error/np.sqrt(self.ntrials)))
        assert error_in_mean<3*mean_error/np.sqrt(self.ntrials),\
                "Error in mean = {}\n\
                 Mean error    = {}".format(
                     error_in_mean, mean_error/np.sqrt(self.ntrials))
        self.mean_sigma = mean_error

    def check_trial_run(self,*args,**kwargs):
        res, err = mcmiser(self.f, self.npoints, self.lbound, self.ubound, *args, **kwargs)
        error_in_mean = abs(res-self.analytical_value)
        assert error_in_mean < 3*self.mean_sigma,\
                "Error in mean  = {}\n\
                 Expected error = {}".format(error_in_mean,self.mean_sigma)

