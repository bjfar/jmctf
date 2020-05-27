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
        self.exact_MLEs = True # Let driver classes know that we can analytically provide exact MLEs, so no numerical fitting is needed.
        
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

        #print("pars:",pars)
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

    def get_Asimov_samples(self,signal_pars):
        """Construct 'Asimov' samples for this analysis
           Used to detemine asymptotic distributions of 
           certain test statistics.
        """
        Asamples = {}
        mu = signal_pars['mu'] * self.mu_scaling
        theta = tf.expand_dims(tf.constant(0,dtype=float),0) # Expand to match shape of signal list 
        Asamples["x"] = tf.expand_dims(mu,0) # Expand to sample dimension size 1
        Asamples["x_theta"] = tf.expand_dims(theta,0) # Expand to sample dimension size 1
        return Asamples

    def get_observed_samples(self):
        """Construct dictionary of observed data for this analysis"""
        Osamples = {}
        Osamples["x"]       = tf.expand_dims(tf.expand_dims(tf.constant(self.x_obs,dtype=float),0),0)
        Osamples["x_theta"] = tf.expand_dims(tf.expand_dims(tf.constant(0,dtype=float),0),0)
        return Osamples

    def get_nuisance_parameter_structure(self):
        """Get a dictionary describing the nuisance parameter structure of this analysis.
           Basically just the keys of the parameter dictionaries plus dimension of each entry"""
        return {"theta": 1} # theta is a scalar

    def get_nuisance_tensorflow_variables(self,sample_dict,signal):
        """Get nuisance parameters to be optimized, for input to "tensorflow_model
           (initial guesses assume fixed signal parameters)
        """
        x = sample_dict["x"]
        x_theta = sample_dict["x_theta"]
        mu = signal['mu'] # non-scaled!
        sigma_t = signal['sigma_t']
        theta_MLE = - ((x - mu)*sigma_t**2 + x_theta*self.sigma**2) / (sigma_t**2 + self.sigma**2)
        thetas = {"theta": tf.Variable(theta_MLE, dtype=float, name='theta')} # Use exact "starting guess", assuming mu is fixed.
        return thetas

    def get_all_tensorflow_variables(self,sample_dict):
        """Get all parameters (signal and nuisance) to be optimized, for input to "tensorflow_model
           (initial guesses assume free 'mu' and 'theta')
        """
        x = sample_dict["x"]
        x_theta = sample_dict["x_theta"]
        pars = {"mu": tf.Variable(x, dtype=float, name='mu'),
                "theta": tf.Variable(x_theta, dtype=float, name='theta')}
        return pars



