"""Analysis class for 'normal' analyses, whose PDF can be described
   by a single Normal distribution.

   This is as simple as it gets, so this class is a good example for
   understanding what information classes derived from BaseAnalysis are 
   required to provide.

   It is, however, a bit trivial since it has no nuisance parameters, so
   it doesn't contribute anything to nuisance parameter fits. For the next
   step up, see NormalTEAnalysis, which has one nuisance parameter.
"""

import numpy as np
import tensorflow as tf
import copy
from tensorflow_probability import distributions as tfd
from .base_analysis import BaseAnalysis
from . import common as c

# Want to convert all this to YAML. Write a simple container to help with this.
class NormalAnalysis(BaseAnalysis):
 
    def __init__(self,name,x_obs,sigma):
        """name  - Name of this analysis
           x_obs - Observed value of measurement
           sigma - 'Baseline' standard deviation. Added in quadrature to pars['sigma'] when constructing tensorflow_model.
        """
        super().__init__(name)

        # Scaling required to make MLE for mu parameter have variance of about 1 (to help out optimiser)
        self.sigma = sigma
        self.mu_scaling = sigma
        self.x_obs = x_obs
        self.exact_MLEs =  True # Let driver classes know that we can analytically provide exact MLEs, so no numerical fitting is needed.

    def tensorflow_model(self,pars):
        """Output tensorflow probability model object, to be combined with models from
           other analysis and sampled from.
           pars - dictionary containing the mean and (additional) standard deviation parameters (tensors, constant or Variable)
           It is assumed that pars['sigma'] is not a free parameter to be fitted, though it may vary with more fundamental
           parameters (could be used for e.g. a theory uncertainty contribution).

           In this simple case the profiling of the nuisance parameter can easily be done analytically,
           however it is not so simple to create a new tensorflow_probability distribution object whose
           log_pdf function would need to contain that result. We can instead just make "perfect" starting
           guesses for the parameter, should be fast enough.
        """

        # Need to construct these shapes to match the event_shape, batch_shape, sample_shape 
        # semantics of tensorflow_probability.
        tfds = {}
        mu = pars['mu'] * self.mu_scaling
 
        # Normal model
        norm = tfd.Normal(loc=mu, scale=self.sigma)

        # Store model in distribution dictionary
        tfds["x"] = norm
        return tfds

    def add_default_nuisance(self,pars):
        """Prepare parameters to be fed to tensorflow model

           Provides default ("nominal") values for nuisance
           parameters if they are not specified.

           Input is full parameter dictionary, with SCALED
           parameters.

           However, there are none for this Analysis class.
           So returns pars unchanged.
        """
        return pars

    def scale_pars(self,pars):
        """Apply scaling (to adjust MLEs to have var~1) to any valid 
        parameters found in input pars dictionary.
        
        NOTE! Will strip out any unrecognised parameters!
        """
        scaled_pars = {}
        if 'mu' in pars.keys():
            scaled_pars['mu'] = pars['mu'] / self.mu_scaling
        return scaled_pars
        
    def descale_pars(self,pars):
        """Remove scaling from parameters. Assumes they have all been scaled and require de-scaling."""
        descaled_pars = {}
        if 'mu' in pars.keys():
            descaled_pars['mu'] = pars['mu'] * self.mu_scaling
        return descaled_pars

    def get_Asimov_samples(self,signal_pars):
        """Construct 'Asimov' samples for this analysis
           Used to detemine asymptotic distributions of 
           certain test statistics.

           Requires unit-scaled parameters as input 
        """
        Asamples = {}
        mu = signal_pars['mu'] * self.mu_scaling
        Asamples["x"] = tf.expand_dims(mu,0) # Expand to sample dimension size 1
        return Asamples

    def get_observed_samples(self):
        """Construct dictionary of observed data for this analysis"""
        Osamples = {}
        Osamples["x"]       = tf.expand_dims(tf.expand_dims(tf.constant(self.x_obs,dtype=c.TFdtype),0),0)
        return Osamples

    def get_interest_parameter_structure(self):
        """Get a dictionary describing the structure of the "interesting" parameters in this analysis
           Basically just the keys of the parameter dictionaries plus dimension of each entry"""
        return {"mu": 1}

    def get_fixed_parameter_structure(self):
        """Get a dictionary describing the structure of the fixed parameters in this analysis
           (i.e. parameters that can be altered along with the signal hypothesis in nuisance
           parameter fits, but which are kept fixed during all fitting.
           Basically just the keys of the parameter dictionaries plus dimension of each entry"""
        return {} # None for this analysis

    def get_nuisance_parameter_structure(self):
        """Get a dictionary describing the nuisance parameter structure of this analysis.
           Basically just the keys of the parameter dictionaries plus dimension of each entry"""
        return {} # None for this analysis

    def get_nuisance_parameters(self,sample_dict,fixed_pars):
        """Get nuisance parameters to be optimized, for input to "tensorflow_model
           (initial guesses assume fixed "signal" parameters)
        """
        mu = fixed_pars['mu'] # non-scaled! 
        free_pars = {} # No nuisance parameters free
        all_fixed_pars = {"mu": mu} # mu is fixed in nuisance-parameter-only fits
        return free_pars, all_fixed_pars

    def get_all_parameters(self,sample_dict,fixed_pars):
        """Get all parameters (signal and nuisance) to be optimized, for input to "tensorflow_model
           (initial guesses assume free 'mu' and 'theta')
           Note that "sigma_t" is an extra theory or control measurement uncertainty
           parameter that cannot be fit, and is always treated as fixed.

           Should return "physical", i.e. non-scaled, initial guesses for parameters
        """
        x = sample_dict["x"]
        mu_MLE = x
        free_pars = {"mu": mu_MLE}
        fixed_pars_out = {} # None
        return free_pars, fixed_pars_out
