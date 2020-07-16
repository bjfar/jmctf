"""Analysis class for 'normal' analyses, whose PDF can be described
   by a single Normal distribution, with a single additive nuisance
   parameter (e.g. could be associated with theory error, or a
   control measurement).

   This is as simple as it gets, so this class is a good example for
   understanding what information classes derived from BaseAnalysis are 
   required to provide.
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
        self.theta_scaling = sigma # Assumes extra model-dependent error will be somewhat similar to sigma
        self.x_obs = x_obs
        self.exact_MLEs =  True # Let driver classes know that we can analytically provide exact MLEs, so no numerical fitting is needed.
        self.const_pars = ['sigma_t']

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
        theta = pars['theta'] * self.theta_scaling 
 
        # Normal models
        norm       = tfd.Normal(loc=mu+theta, scale=self.sigma) # TODO: shapes probably need adjustment
        norm_theta = tfd.Normal(loc=theta, scale=pars['sigma_t'])

        # Store model in distribution dictionary
        # Naming is import for (TODO: can't remember?)
        tfds["x"] = norm
        tfds["x_theta"] = norm_theta
        return tfds

    def add_default_nuisance(self,pars):
        """Prepare parameters to be fed to tensorflow model

           Provides default ("nominal") values for nuisance
           parameters if they are not specified.

           Input is full parameter dictionary, with SCALED
           parameters
        """
         
        pars_out = {}
        if 'theta' not in pars.keys():
            # trigger shortcut to set nuisance parameters to zero. Useful for sample generation. 
            theta = tf.constant(0*pars['mu'])
        else:
            theta = pars['theta']

        if 'sigma_t' not in pars.keys():
            # Default for when no extra "theory" uncertainty is provided
            sigma_t = tf.constant([c.reallysmall],dtype=c.TFdtype) # Cannot use exactly zero, gets nan from TF due to zero width normal dist. Use something "near" smallest positive 32-bit float instead.
        else:
            sigma_t = pars['sigma_t'] 

        pars_out['mu'] = pars['mu']
        pars_out['sigma_t'] = sigma_t # Always treated as fixed, so no scaling ever needed
        pars_out['theta'] = theta
        return pars_out

    def scale_pars(self,pars):
        """Apply scaling (to adjust MLEs to have var~1) to any valid 
        parameters found in input pars dictionary.
        
        NOTE! Will strip out any unrecognised parameters!
        """
        scaled_pars = {}
        if 'mu' in pars.keys():
            scaled_pars['mu'] = pars['mu'] / self.mu_scaling
        if 'theta' in pars.keys(): 
            scaled_pars['theta'] = pars['theta'] / self.theta_scaling
        if 'sigma_t' in pars.keys():
            scaled_pars['sigma_t'] = pars['sigma_t'] # No scaling; always fixed.
        return scaled_pars
        
    def descale_pars(self,pars):
        """Remove scaling from parameters. Assumes they have all been scaled and require de-scaling."""
        descaled_pars = {}
        if 'mu' in pars.keys():
            descaled_pars['mu'] = pars['mu'] * self.mu_scaling
        if 'theta' in pars.keys(): 
            descaled_pars['theta'] = pars['theta'] * self.theta_scaling
        if 'sigma_t' in pars.keys():
            descaled_pars['sigma_t'] = pars['sigma_t'] # No scaling; always fixed.
        return descaled_pars

    def get_Asimov_samples(self,signal_pars):
        """Construct 'Asimov' samples for this analysis
           Used to detemine asymptotic distributions of 
           certain test statistics.

           Requires unit-scaled parameters as input 
        """
        Asamples = {}
        mu = signal_pars['mu'] * self.mu_scaling
        theta = tf.expand_dims(tf.constant(0,dtype=c.TFdtype),0) # Expand to match shape of signal list 
        Asamples["x"] = tf.expand_dims(mu,0) # Expand to sample dimension size 1
        Asamples["x_theta"] = tf.expand_dims(theta,0) # Expand to sample dimension size 1
        return Asamples

    def get_observed_samples(self):
        """Construct dictionary of observed data for this analysis"""
        Osamples = {}
        Osamples["x"]       = tf.expand_dims(tf.expand_dims(tf.constant(self.x_obs,dtype=c.TFdtype),0),0)
        Osamples["x_theta"] = tf.expand_dims(tf.expand_dims(tf.constant(0,dtype=c.TFdtype),0),0)
        return Osamples

    def get_free_parameter_structure(self):
        """Get a dictionary describing the structure of the free parameters in this analysis
           Basically just the keys of the parameter dictionaries plus dimension of each entry"""
        return {"mu": 1}

    def get_fixed_parameter_structure(self):
        """Get a dictionary describing the structure of the fixed parameters in this analysis
           (i.e. parameters that can be altered along with the signal hypothesis in nuisance
           parameter fits, but which are kept fixed during all fitting.
           Basically just the keys of the parameter dictionaries plus dimension of each entry"""
        return {"sigma_t": 1}

    def get_nuisance_parameter_structure(self):
        """Get a dictionary describing the nuisance parameter structure of this analysis.
           Basically just the keys of the parameter dictionaries plus dimension of each entry"""
        return {"theta": 1} # theta is a scalar

    def get_nuisance_parameters(self,sample_dict,fixed_pars):
        """Get nuisance parameters to be optimized, for input to "tensorflow_model
           (initial guesses assume fixed "signal" parameters)
        """
        x = sample_dict["x"]
        x_theta = sample_dict["x_theta"]
        mu = fixed_pars['mu'] # non-scaled!
        if 'sigma_t' in fixed_pars.keys():
            sigma_t = fixed_pars['sigma_t']
        else:
            sigma_t = c.reallysmall # TODO: Cannot use exactly zero 
        theta_MLE = ((x - mu)*sigma_t**2 + x_theta*self.sigma**2) / (sigma_t**2 + self.sigma**2)
        free_pars = {"theta": theta_MLE} # Use exact "starting guess", assuming mu is fixed.
        all_fixed_pars = {"mu": mu, "sigma_t": sigma_t} # mu is fixed in nuisance-parameter-only fits

        # chi2_x = (mu + theta_MLE - x)**2 / self.sigma**2
        # chi2_xt = (theta_MLE - x_theta)**2 / sigma_t**2
        # #print("chi2_x:", chi2_x)
        # #print("chi2_xt:", chi2_xt)
        # #print("chi2:", chi2_x + chi2_xt)
        # const_x = np.log(2*np.pi) + 2*np.log(self.sigma)
        # const_xt = np.log(2*np.pi) + 2*np.log(sigma_t)
        # #print("-2logL_x:", chi2_x + const_x)
        # #print("-2logL_xt:", chi2_xt + const_xt)
        # #print("-2logL:", chi2_x + chi2_xt + const_x + const_xt)
        return free_pars, all_fixed_pars

    def get_all_parameters(self,sample_dict,fixed_pars):
        """Get all parameters (signal and nuisance) to be optimized, for input to "tensorflow_model
           (initial guesses assume free 'mu' and 'theta')
           Note that "sigma_t" is an extra theory or control measurement uncertainty
           parameter that cannot be fit, and is always treated as fixed.

           Should return "physical", i.e. non-scaled, initial guesses for parameters
        """
        x = sample_dict["x"]
        x_theta = sample_dict["x_theta"]
        mu_MLE = x - x_theta
        theta_MLE = x_theta
        free_pars = {"mu": mu_MLE, "theta": theta_MLE} 
        if 'sigma_t' in fixed_pars.keys():
            sigma_t = fixed_pars['sigma_t']
        else:
            sigma_t = c.reallysmall # Default TODO: Cannot use exactly zero 
        fixed_pars_out = {"sigma_t": sigma_t}
        
        # chi2_x = (mu_MLE + theta_MLE - x)**2 / self.sigma**2
        # chi2_xt = (theta_MLE - x_theta)**2 / sigma_t.numpy()**2
        # #print("chi2_x:", chi2_x)
        # #print("chi2_xt:", chi2_xt)
        # #print("chi2:", chi2_x + chi2_xt)
        # const_x = np.log(2*np.pi) + 2*np.log(self.sigma)
        # const_xt = np.log(2*np.pi) + 2*np.log(sigma_t.numpy())
        # #print("-2logL_x:", chi2_x + const_x)
        # #print("-2logL_xt:", chi2_xt + const_xt)
        # #print("-2logL:", chi2_x + chi2_xt + const_x + const_xt)
        return free_pars, fixed_pars_out



