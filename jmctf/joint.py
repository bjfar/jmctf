"""Classes and functions associated with creating and fitting joint distribution objects"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from collections.abc import Mapping
import contextlib
import time
import functools
import massminimize as mm
from . import common as c

import traceback

#tmp
id_only = False

#===================
# Helper functions for optimizer test replacement
# From examples at https://www.tensorflow.org/probability/examples/Optimizers_in_TensorFlow_Probability
# ==================
def make_val_and_grad_fn(value_fn):
  @functools.wraps(value_fn)
  def val_and_grad(x):
    return tfp.math.value_and_gradient(value_fn, x)
  return val_and_grad

def np_value(tensor):
  """Get numpy value out of possibly nested tuple of tensors."""
  if isinstance(tensor, tuple):
    return type(tensor)(*(np_value(t) for t in tensor))
  else:
    return tensor.numpy()

def run(optimizer):
  """Run an optimizer and measure it's evaluation time."""
  optimizer()  # Warmup.
  with timed_execution():
    result = optimizer()
  return np_value(result)

@contextlib.contextmanager
def timed_execution():
  t0 = time.time()
  yield
  dt = time.time() - t0
  print('Evaluation took: %f seconds' % dt)
#=================

def combine_pars(pars1,pars2=None):
    """Combine parameter dictionaries, with first argument taking precedence"""
    if pars2 is None:
        all_pars = pars1
    else:
        all_pars = c.deep_merge(pars2,pars1) # Second argument takes precendence in deep_merge
    return all_pars

def neg2logL(pars,const_pars,analyses,data,transform=None):
    """General -2logL function to optimise
       TODO: parameter 'transform' feature not currently in use, probably doesn't work correctly
    """
    #print("In neg2logL:")
    #print("In neg2logL: pars (scaled) = ", pars)
    #print("In neg2logL: const_pars (scaled) = ", const_pars)
    #print("pars:", c.print_with_id(pars,id_only))
    #print("const_pars:", c.print_with_id(const_pars,id_only))
    if transform is not None:
        pars_t = transform(pars)
    else:
        pars_t = pars
    all_pars = combine_pars(pars_t, const_pars)

    #tf.print("pars:", pars)
    #tf.print("all_pars:", all_pars)

    # Sanity check: make sure parameters haven't become nan somehow
    # Doesn't seem to work with @tf.function decorator or inside tfp optimizer
    # ----
    # anynan = False
    # nanpar = ""
    # for a,par_dict in pars.items():
    #     for p, val in par_dict.items():
    #         if tf.math.reduce_any(tf.math.is_nan(val)):
    #             anynan = True
    #             nanpar += "\n    {0}::{1}".format(a,p)

    # if anynan:
    #     msg = "NaNs detected in parameter arrays during optimization! The fit may have become unstable and wandering into an invalid region of parameter space; please check your analysis setup. Parameter arrays containing NaNs were:{0}".format(nanpar)
    #     tf.print("pars:", pars)
    #     raise ValueError(msg)
    # ----

    # Parameters will enter this function pre-scaled such that MLEs have variance ~1
    # So we need to set the pre-scaled flag for the JointDistribution constructor to
    # avoid applying the scaling a second time.
    joint = JointDistribution(analyses.values(),all_pars,pre_scaled_pars=True,in_tf_function=True)
    q = -2*joint.log_prob(data)
    #print("in neg2logL: all_pars = ", joint.descale_pars(all_pars))
    #print("in neg2logL: joint.get_pars() = ",joint.get_pars())
    #print("q:", q)
    #print("all_pars:", all_pars)
    #print("logL parts:", joint.log_prob_parts(data))

    # Can't do this in @tf.function either
    # if tf.math.reduce_any(tf.math.is_nan(q)):
    #     # Attempt to locate components generating the nans
    #     component_logprobs = joint.log_prob_parts(data)
    #     nan_components = ""
    #     for comp,val in component_logprobs.items():
    #         if tf.math.reduce_any(tf.math.is_nan(val)):
    #             nan_components += "\n    {0}".format(comp)                
    #     msg = "NaNs detect in result of neg2logL calculation! Please check that your input parameters are valid for the distributions you are investigating, and that the fit is stable! Components of the joint distribution whose log_prob contained nans were:" + nan_components
    #     raise ValueError(msg)
    total_loss = tf.math.reduce_sum(q)
    #print("all_pars:", all_pars)
    #print("joint.descale_pars(all_pars):", joint.descale_pars(all_pars))
    #quit()
    return total_loss, q, joint.descale_pars(all_pars), None

def optimize(pars,const_pars,analyses,data,transform=None,log_tag='',verbose=False,force_numerical=False):
    """Wrapper for optimizer step that skips it if the initial guesses are known
       to be exact MLEs"""
    opts = {"optimizer": "Adam",
            "step": 0.05,
            "tol": 0.01,
            "grad_tol": 1e-4,
            "max_it": 100,
            "max_same": 5,
            "log_tag": log_tag,
            "verbose": verbose 
            }

    kwargs = {'analyses': analyses,
              'data': data,
              'transform': transform
              }
    #print("In 'optimize'")
    #print("pars:", c.print_with_id(pars,id_only))
    #print("const_pars:", c.print_with_id(const_pars,id_only))

    # Convert free parameter initial guesses into TensorFlow Variable objects
    free_pars = c.convert_to_TF_variables(pars) 

    #print("Converted free parameters into TensorFlow Variables:")
    #print("free_pars:", free_pars)
    #print("free_pars (id):", c.print_with_id(free_pars,id_only))

    # Sanity check input parameters
    anynan = False
    nanpar = ""
    for pars_tf in [pars, const_pars]:
        for a,par_dict in pars_tf.items():
            for p, val in par_dict.items():
                if tf.math.reduce_any(tf.math.is_nan(val)):
                    anynan = True
                    nanpar += "\n    {0}::{1}".format(a,p)
    if anynan:
        msg = "NaNs detected in input parameter arrays for 'optimize' function! Parameter arrays containing NaNs were:{0}".format(nanpar)
        raise ValueError(msg)

    if force_numerical:
        all_exact_MLEs = False
    else:
        all_exact_MLEs = True
        for a in analyses.values():
            if not a.exact_MLEs: all_exact_MLEs = False

    if all_exact_MLEs:
        if verbose: print("All starting MLE guesses are exact: skipping optimisation") 
        total_loss, q, final_pars, null = neg2logL(free_pars,const_pars,**kwargs)
        #print("Finished using exact MLEs: final_pars = ", final_pars)
    else:
        # For analyses that have exact MLEs, we want to move those parameters from the
        # "free" category into the "fixed" category.
        reduced_free_pars = {}
        enlarged_const_pars = {}
        for a in analyses.values():
            a_free_pars  = free_pars.get(a.name,{})
            a_const_pars = const_pars.get(a.name,{})
            if a.exact_MLEs:
                # Move free parameters for this analysis into "const" category
                if verbose: print("Starting MLE guesses are exact for analysis '{0}', removing these parameters from optimisation step.".format(a.name))
                enlarged_const_pars[a.name] = c.deep_merge(a_free_pars,a_const_pars)
            else:
                # Keep free parameters for this analysis in the "free" category
                reduced_free_pars[a.name] = a_free_pars
                enlarged_const_pars[a.name] = a_const_pars

        if verbose: print("Beginning optimisation")
        #f = tf.function(mm.tools.func_partial(neg2logL,**kwargs))
        kwargs["const_pars"] = enlarged_const_pars

        # Optimization with massminimize
        # -------------------
        f = mm.tools.func_partial(neg2logL,**kwargs)
        #print("About to enter optimizer")
        #print("pars:", c.print_with_id(reduced_free_pars,False))
        q, final_pars, null = mm.optimize(reduced_free_pars, f, **opts)
        #print("Finished mm.optimize: final_pars = ", final_pars)
        # ------------------

        # # Optimization with tfp optimizers
        # # So far doesn't work. No idea why, the
        # # errors are completely opaque.
        # # --------------------

        # # Stack initial parameter guesses into single tensor
        # start_point = c.cat_pars(reduced_free_pars)

        # # Slightly altered function signature required
        # # Need to de-stack single tensor back into parameter dictionary
        # @make_val_and_grad_fn
        # def neg2logL_tfp(catted_pars):
        #     pars = c.uncat_pars(catted_pars, pars_template=reduced_free_pars)
        #     total_loss, q, x1, x2 = neg2logL(pars,**kwargs)
        #     return q

        # # Function to run optimizer
        # @tf.function
        # def optimizer():
        #    return tfp.optimizer.lbfgs_minimize(
        #       neg2logL_tfp, initial_position=start_point,
        #       stopping_condition=tfp.optimizer.converged_all,
        #       max_iterations=100,
        #       tolerance=1e-8)
 
        # results = run(optimizer)
        # print("results:", results)
        # quit()
        # # --------------------

    # Rebuild distribution object with fitted parameters for output to user
    joint = JointDistribution(analyses.values(),final_pars)

    # Split parameters back into fitted vs const parameters
    # (as the user saw them; i.e. undoing the "reduced" free pars stuff in exact MLE case)
    # 'const' parameters can occur by default, so the "free" parameters take precedence when clashes exist
    final_free_pars = {ak: {p: final_pars[ak][p] for p in av.keys()} for ak,av in pars.items()}
    final_const_pars = {ak: {p: final_pars[ak][p] for p in av.keys() if p not in pars.get(ak,{}).keys()} for ak,av in const_pars.items()}
 
    #print("final_pars:", final_pars)
    #print("pars:", pars)
    #print("final_free_pars:", final_free_pars)
    #print("const_pars:", const_pars)
    #print("final_const_pars:", final_const_pars)

    # Output is:
    #  JointDistribution with parameters set to the MLEs,
    #  -2*log_prob(samples) of the JointDistribution under the MLEs for all samples
    #  Full parameter dictionary used to construct the fitted JointDistribution
    #  parameter dictionary containing only the fitted free parameters
    #  parameter dictionary containing only the fixed ("bystander") parameters
    return joint, q, final_pars, final_free_pars, final_const_pars

class JointDistribution(tfd.JointDistributionNamed):
    """Object to combine analyses together and treat them as a single
       joint distribution. Uses JointDistributionNamed for most of the
       underlying work.

       TODO: This object has a bunch of stuff that only works with BinnedAnalysis
             objects as the 'analyses'. Needs to be generalised. 
    """
   
    def __init__(self, analyses, pars=None, pre_scaled_pars=False, in_tf_function=False):
        """ 
        :param analyses: list of analysis-like objects to be combined
        :type analyses: list
        :param pars: dictionary of parameters for all analysis objects, to fix
                parameters for sampling (default: None)
        :type pars: dictionary, optional
        :param pre_scaled_pars: If True, all input parameters are already scaled 
                such that MLEs have variance of approx. 1 (for more stable fitting).
                If False, all parameters are conventionally (i.e. not) scaled, and
                require scaling internally.
        :type pre_scaled_pars: bool, optional
        :param in_tf_function: If True, assumes this object is being used inside a tf.function.
                Disables certain operations that cannot run inside a tf.function, for example
                certain boolean checks and exception handling. Basically removes some sanity
                checking etc.
        :type in_tf_function: bool, optional
        """
        #print("In JointDistribution constructor (pre_scaled_pars={0})".format(pre_scaled_pars))
         
        self.analyses = {a.name: a for a in analyses}
        self.Osamples = {}
        for a in self.analyses.values():
            self.Osamples.update(c.add_prefix(a.name,a.get_observed_samples()))
        if pars is not None:
            # Convert parameters to TensorFlow constants, if not already TensorFlow objects
            #print("pars:", c.print_with_id(pars,id_only))
            pars_tf = c.convert_to_TF_constants(pars,ignore_variables=True)
            #print("pars_tf:", c.print_with_id(pars_tf,id_only))

            if not in_tf_function:
                # Check that parameters are not NaN
                anynan = False
                nanpar = ""
                for a,par_dict in pars_tf.items():
                    for p, val in par_dict.items():
                        if tf.math.reduce_any(tf.math.is_nan(val)):
                            anynan = True
                            nanpar += "\n    {0}::{1}".format(a,p)
                if anynan:
                    msg = "NaNs detected in input parameter arrays for JointDistribution! Parameter arrays containing NaNs were:{0}".format(nanpar)
                    raise ValueError(msg)
            self.pars = self.prepare_pars(pars_tf,pre_scaled_pars)
            #print("JointDistribution constructor: self.pars:", c.print_with_id(self.pars,id_only))

            #-----------
            # Manually print call stack to see who created this object
            #traceback.print_stack()
            #-----------

            dists = {} 
            self.Asamples = {}
            for a in self.analyses.values():
                d = c.add_prefix(a.name,a.tensorflow_model(self.pars[a.name]))
                dists.update(d)
                self.Asamples.update(c.add_prefix(a.name,a.get_Asimov_samples(self.pars[a.name])))
            super().__init__(dists) # Doesn't like it if I use self.dists, maybe some construction order issue...
            self.dists = dists

            # Check that a consistent batch_shape can be found!
            # Will throw an error if it cannot.
            batch_shape = self.bcast_batch_shape_tensor()
        else:
            self.pars = None
        # If no pars provided can still fit the analyses, but obviously cannot sample or compute log_prob etc.
        # TODO: can we fail more gracefully if people try to do this?
        #       Or possibly the fitting stuff should be in a different object? It seems kind of nice here though.
        #print("self.pars = ", self.pars)

    def has_pars(self):
        """Determine whether parameters have been provided to this object.
           If they haven't then the underlying JointDistributionNamed object has
           not been initialised, so certain methods cannot be called"""
        return self.pars != None

    def identify_const_parameters(self):
        """Ask component analyses to report which of their parameters are to be
           considered as always "constant", when it comes to computing gradients with 
           respect to the log_prob (especially Hessians)."""
        const_pars = {}
        for a in self.analyses.values():
            try:
                const_pars[a.name] = a.const_pars
            except AttributeError:
                const_pars[a.name] = []
        return const_pars

    def fix_parameters(self, pars):
       """Return a version of this JointDistribution object that has parameters fixed to the supplied values"""
       return JointDistribution(self.analyses.values(), pars)

    def biased_sample(self, N, bias=1):
       """Sample from biased versions of all analyses and return them along their with sampling probability. For use in importance sampling.
        
       :param N: Number of samples to draw
       :type N: int
       :param bias: indicates how many 'sigma' of upward bias to apply to the sample 
               generation, computed, in terms of sqrt(variance) of the background.
               Bias only applied to 'signal' parameters, not nuisance parameters.
               NOTE: This doesn't really work properly. Importance sampling is a bit tricky, 
               probably need a smarter way of choosing the 'importance' distribution. (default value=1)
       :type bias: float, optional
       """
       biased_analyses = copy.deepcopy(self.analyses)
       for a in biased_analyses.values():
           a.SR_b = a.SR_b + bias*np.sqrt(a.SR_b)
       biased_joint = JointDistribution(biased_analyses.values(), self.pars, pre_scaled_pars=True)
       samples = biased_joint.sample(N)
       logw = self.log_prob(samples) - biased_joint.log_prob(samples) # log(weight) for each sample
       return samples, logw

    def prepare_pars(self,pars,pre_scaled_pars=False):
        """Prepare default nuisance parameters and return scaled signal and nuisance parameters for each analysis
           (scaled such that MLE's in this parameterisation have
           variance of approx. 1"""

        # First check whether all expected parameters are found
        for a in self.analyses.values():
            if a.name not in pars.keys(): raise KeyError("Parameters for analysis {0} not found!".format(a.name))
 
        all_pars_1 = {}
        # Next check whether parameters are pre-scaled and whether they are already TensorFlow objects
        for a in self.analyses.values():
            # It is an error to scale TensorFlow 'Variable' objects! These
            # should not be used as input unless it is occurring internally in
            # the TensorFlow optimizer. In which case the parameters should already
            # be scaled.
            if pre_scaled_pars:
                p = pars[a.name]
            else:
                try:
                    p = c.convert_to_TF_constants(pars[a.name],ignore_variables=False)
                except TypeError as e:
                    msg = "TensorFlow 'Variable' objects found in the input parameter dictionary for analysis {0}, but parameters are not flagged as 'pre-scaled'! Please do not use this type for input to JointDistribution parameters, as it is reserved for internal use with the TensorFlow optimizer routines, and needs to be controlled to maintain the correct graph relationships between input parameters and the log_prob output of the JointDistribution. Any other list/tuple/array type structure should be used instead."  
                    raise TypeError(msg) from e
            all_pars_1[a.name] = p

        # Next, add the nuisance parameters in if needed, and check their shapes too
        # (but this time the squeezing of axis=0 only occurs if it did for the user parameters)
        all_pars_4 = {}
        for a in self.analyses.values():
            all_pars_4[a.name] = a.add_default_nuisance(all_pars_1[a.name])

        # Finally, do scaling if needed
        all_pars_out = {}
        if not pre_scaled_pars:
            for a in self.analyses.values():
                all_pars_out[a.name] = a.scale_pars(all_pars_4[a.name])
        else:
            all_pars_out = all_pars_4

        # Throw warning about discarded parameters, in case user messed up the input
        for a in self.analyses.values():
            missing = []
            for par in pars[a.name].keys():
                if par not in p.keys():
                    missing += [par]
            if len(missing)>0:
                msg = "***WARNING: the following unrecognised parameters were found in the parameter dictionary for analysis {0}: {1}\nThis is permitted, but please make sure it wasn't an accident.".format(a.name, missing)
                print(msg)

        return all_pars_out

    def descale_pars(self,pars):
        """Remove scaling from parameters. Assumes they have all been scaled and require de-scaling."""
        descaled_pars = {}
        for a in self.analyses.values():
          if a.name in pars.keys():
            descaled_pars[a.name] = a.descale_pars(pars[a.name])
        return descaled_pars

    def get_pars(self):
        """Return all the parameters of the distribution, in "physical" scaling."""
        return self.descale_pars(self.pars)

    def scale_pars(self,pars):
        """Apply scaling to all parameters. Assume none of them have had scaling applied yet."""
        scaled_pars = {}
        for a in self.analyses.values():
          if a.name in pars.keys():
            scaled_pars[a.name] = a.scale_pars(pars[a.name])
        return scaled_pars

    def get_nuis_parameters(self,samples,fixed_pars):
        """Samples vector and signal provided to compute good starting guesses for parameters
           (in scaled parameter space)"""
        pars = {}
        all_fixed_pars = {}
        for a in self.analyses.values():
            if a.name not in fixed_pars:
                msg = "No fixed parameters supplied for analysis {0} during nuisance parameter fit! To fit only the nuisance parameters, fixed values for all non-nuisance parameters need to be provided".format(a.name)
                raise ValueError(msg)
            # Get samples and parameters for this analysis and broadcast them against each other
            bcast_pars, bcast_samples = a.bcast_parameters_samples(fixed_pars[a.name],self.get_samples_for(a.name,samples))
            p, fp = a.get_nuisance_parameters(bcast_samples,bcast_pars)
            # Apply scaling to all parameters, so that scan occurs in ~unit scale parameter space
            pars[a.name] = a.scale_pars(p)
            all_fixed_pars[a.name] = a.scale_pars(fp)
        #print("pars:", c.print_with_id(pars,id_only))
        #print("fixed_pars:", c.print_with_id(fixed_pars,id_only))
        return pars, all_fixed_pars

    def get_samples_for(self,name,samples):
        """Extract the samples for a specific analysis from a sample dictionary, and
           remove the analysis name prefix from the keys"""
        d = {key:val for key,val in samples.items() if key.startswith("{0}::".format(name))}
        return c.remove_prefix(name,d)

    def get_all_parameters(self,samples,fixed_pars={}):
        """Samples vector and signal provided to compute good starting guesses for parameters
           (in scaled parameter space)"""
        pars = {}
        all_fixed_pars = {}
        anynan = False
        nanpar = ""
        # TODO: Add error checking for analysis names in fixed_pars dict? But could be useful to allow
        # "extra" analyses to be in there. Perhaps make check optional via a flag (default on)?
        for a in self.analyses.values():
            p, fp = a.get_all_parameters(self.get_samples_for(a.name,samples), fixed_pars.get(a.name,{}))
            # Check the starting guesses are valid
            for pardicts_in in [p,fp]: 
                for par, val in pardicts_in.items():
                    if tf.math.reduce_any(tf.math.is_nan(val)):
                        anynan = True
                        nanpar += "\n    {0}::{1}".format(a.name,par)
            # Apply scaling to all parameters, so that scan occurs in ~unit scale parameter space
            pars[a.name] = a.scale_pars(p)
            all_fixed_pars[a.name] = a.scale_pars(fp)
        if anynan:
            msg = "NaNs detected in parameter starting guesses! The samples used to inform the starting guesses may be invalid (e.g. negative counts for Poisson variables). Parameter starting guess arrays containing NaNs were:{0}".format(nanpar)
            raise ValueError(msg)
        return pars, fixed_pars

    def fit_nuisance(self,samples,fixed_pars=None,log_tag='',verbose=False,force_numeric=False):
        """Fit nuisance parameters to samples for a fixed signal
           (ignores parameters that were used to construct this object).
           If force_numeric is True then asserted 'exactness' of starting guesses
           is ignored and numerical optimisation is run regardless."""
        if fixed_pars is None:
            fixed_pars = self.get_pars() # Assume hypotheses provided at construction time (but need de-scaled parameters here!)
        print("fixed_pars:", fixed_pars)
        fp = c.convert_to_TF_constants(fixed_pars)
        all_nuis_pars, all_fixed_pars = self.get_nuis_parameters(samples,fp)
        print("all_nuis_pars:", all_nuis_pars)
        print("all_fixed_pars:", all_fixed_pars)
        print("samples:", samples)

        # Note, parameters obtained from get_nuis_parameters, and passed to
        # the 'optimize' function, are SCALED. All of them, regardless of whether
        # they actually vary in this instance.
        joint_fitted, q, all_pars, fitted_pars, const_pars = optimize(all_nuis_pars,all_fixed_pars,self.analyses,samples,log_tag=log_tag,verbose=verbose,force_numerical=force_numeric)

        # Fitted/final parameters are returned de-scaled
        # Also it is nice to pack up the various parameter splits into a dictionary
        par_dict = {}
        par_dict["all"]    = all_pars
        par_dict["fitted"] = fitted_pars
        par_dict["fixed"]  = const_pars
        return -0.5*q, joint_fitted, par_dict 


    # TODO: Deprecated, but may need something like this again.
    #def fit_nuisance_and_scale(self,signal,samples,log_tag='',verbose=False):
    #    """Fit nuisance parameters plus a signal scaling parameter
    #       (ignores parameters that were used to construct this object)"""
    #    pars = self.get_nuis_parameters(signal,samples)
    #    # Signal scaling parameter. One per sample, and per signal input
    #    Nsamples = list(samples.values())[0].shape[0]
    #    Nsignals = list(list(signal.values())[0].values())[0].shape[0] 
    #    muV = tf.Variable(np.zeros((Nsamples,Nsignals,1)),dtype=c.TFdtype)
    #    # Function to produce signal parameters from mu
    #    def mu_to_sig(pars):
    #        mu = tf.sinh(pars['mu']) # sinh^-1 is kind of like log, but stretches similarly for negative values. Seems to be pretty good for this.
    #        sig_out = {}
    #        for ka,a in signal.items():
    #            sig_out[ka] = {}
    #            for kp,p in a.items():
    #                sig_out[ka][kp] = mu*p
    #        nuis_pars = {k:v for k,v in pars.items() if k is not 'mu'}
    #        out = c.deep_merge(nuis_pars,sig_out) # Return all signal and nuisance parameters, but not mu.
    #        return out 
    #    pars['mu'] = muV # TODO: Not attached to any analysis, but might work anyway  
    #    joint_fitted, q = optimize(pars,None,self.analyses,samples,pre_scaled_pars='nuis',transform=mu_to_sig,log_tag=log_tag,verbose=verbose)
    #    return q, joint_fitted, pars
  
    def fit_all(self,samples,fixed_pars=None,log_tag='',verbose=False,force_numeric=False):
        """Fit all signal and nuisance parameters to samples
           (ignores parameters that were used to construct this object)
           Some special parameters within analyses are also flagged as
           un-fittable, e.g. theory uncertainty parameters. If these
           aren't provided then analyses will use default fixed values,
           but they can be supplied via the "fixed_pars" dict. These
           parameters are always treated as fixed, when it comes to
           starting MLE guesses etc.
           If force_numeric is True then asserted 'exactness' of starting guesses
           is ignored and numerical optimisation is run regardless.
        """
        if fixed_pars is None:
            fixed_pars = self.get_pars() # Assume any extra fixed parameters were provided at construction time. If missing defaults will be used.

        # Make sure the samples are TensorFlow objects of the right type:
        samples = {k: tf.constant(x,dtype="float32") for k,x in samples.items()}
        fp = c.convert_to_TF_constants(fixed_pars)
        all_free_pars, all_fixed_pars = self.get_all_parameters(samples,fp)

        # Note, parameters obtained from get_all_parameters, and passed to
        # the 'optimize' function, are SCALED. All of them, regardless of whether
        # they actually vary in this instance.
        joint_fitted, q, all_pars, fitted_pars, const_pars = optimize(all_free_pars,all_fixed_pars,self.analyses,samples,log_tag=log_tag,verbose=verbose,force_numerical=force_numeric)

        # Fitted/final parameters are returned de-scaled
        # Also it is nice to pack up the various parameter splits into a dictionary
        par_dict = {}
        par_dict["all"]    = all_pars
        par_dict["fitted"] = fitted_pars
        par_dict["fixed"]  = const_pars
        return -0.5*q, joint_fitted, par_dict 

    def get_best_fit(self,samples):
        """Based on parameters belonging to this object, return the parameters that result
           in the highest log_prob value for the given input samples.
           Mainly intended to be used after fitting multiple hypotheses to the same samples.
        """
        log_probs = self.log_prob(samples)


    def Hessian(self,samples):
        """Obtain Hessian matrix (and grad) of the log_prob function at 
           input parameter points
           Make sure to use de-scaled parameters as input!

           Shape requirements: 
            samples should match self.pars, in the sense that the 
            sample_dims+batch_dims for the samples should match the
            batch_dims that results from self.pars, after broadcasting.
            That is, the samples should be matched one-to-one to (broadcast) 
            parameters.

            (e.g. we want one Hessian per sample, expanded around
            some parameter point tailored to each sample).

           Parameters should be those already known internally to this
           JointDistribution.

           Output shape will be (batch_dims,N,N), where N is the number of
           scalar parameters in the joint distribution (i.e. after parameter
           flattening)
        """

        # Check batch shapes associated with internal probability models
        # This will affect the output Hessian shape, i.e. we 
        # Returns a dict of batch shapes.
        batch_shape = self.bcast_batch_shape_tensor()
        #print("Hessian: batch_shape:", batch_shape)

        # Make sure to use non-scaled parameters to get correct gradients etc.
        pars = self.get_pars()

        #print("self.pars:", pars)

        # Separate "const" parameters
        free_pars = {}
        const_pars = {}
        const_par_names = self.identify_const_parameters()
        for a,p in pars.items():
            free_pars[a] = {}
            const_pars[a] = {}
            for name,v in p.items():
                if name in const_par_names[a]:
                    const_pars[a][name] = v
                else:
                    free_pars[a][name] = v

        # Stack parameters into a single tensorflow variable for
        # matrix manipulations
        par_shapes = self.parameter_shapes()
        all_input_pars, bcast_batch_shape, column_names = c.cat_pars_to_tensor(free_pars,par_shapes)
        npars = len(column_names)
        #print("pars:",pars)
        #print("column_names:", column_names)

        if bcast_batch_shape != batch_shape:
            msg = "Broadcasted batch shape inferred while stacking parameters into tensor did not match batch shape inferred from underlying distribution objects! This is a bug, if there is a problem with the input parameters it should have been detected before this."
            raise ValueError(msg)

        input_pars = tf.Variable(all_input_pars)
        with tf.GradientTape() as tape_outer:
            with tf.GradientTape() as tape:
            #with tf.GradientTape(persistent=True,watch_accessed_variables=False) as tape:
                tape.watch(input_pars)
                # Don't need to go via JointDistribution, can just
                # get log_prob for all component dists "manually"
                # Avoids confusion about parameters getting copied and
                # breaking TF graph connections etc.
                #print("samples:", samples)
                #print("free_pars:", free_pars)
                #print("input_pars:", input_pars)
                #print("const_pars:", const_pars)
                #print("catted_pars:", catted_pars)
                inpars = c.decat_tensor_to_pars(input_pars,free_pars,par_shapes,batch_shape) # need to unstack for use in each analysis
         
                # print("inpars:", inpars)
                # merge with const parameters
                all_inpars = c.deep_merge(inpars,const_pars)
                scaled_inpars = self.scale_pars(all_inpars)
                q = 0
                for a in self.analyses.values():
                    d = c.add_prefix(a.name,a.tensorflow_model(scaled_inpars[a.name])) 
                    for dist_name, dist in d.items():
                        #print("x[{0}]:".format(dist_name), samples[dist_name])
                        #print("dist {0} description: {1}".format(dist_name,dist))
                        q += dist.log_prob(samples[dist_name])
                #print("q:", q)
            grads = tape.gradient(q, input_pars) #[0]
            #grads = tape.jacobian(q, catted_pars)
            #grads = tape.batch_jacobian(q, catted_pars)
            #grads = tape.hessians(q, catted_pars)
            #print("samples:", samples)
            #print("all_inpars:", all_inpars)
            # print("scaled_inpars:", scaled_inpars)
            # print("catted_pars:", catted_pars)
            #print("input_pars:", input_pars)
            #print("q:", q)
            #print("grads:", grads)
        # Compute Hessians. batch_jacobian takes first (the sample) dimensions as independent for much better efficiency,
        hessians = tape_outer.batch_jacobian(grads, input_pars) 
        #...but we are only allowing one sample anyway, so can just do normal jacobian
        #hessian = tape_outer.jacobian(grads, input_pars)
        #print("H:",hessians)

        # # If Hessian (or grad) dimension is too large (due to extra singleton dimensions in either the samples or the
        # # input parameters) then squeeze them out until we get to the right shape.
        # hessians_out = c.squeeze_to(hessians,d=3,dont_squeeze=[0])
        # grads_out = c.squeeze_to(grads,d=2,dont_squeeze=[0])
        # print("g_out:",grads_out)
        # print("H_out:",hessians_out)

        # Reshape to restore the batch dimensions
        h_out_shape = [d for d in batch_shape] + [npars,npars]
        g_out_shape = [d for d in batch_shape] + [npars]
        hessians_out = tf.reshape(hessians,h_out_shape)
        grads_out = tf.reshape(grads,g_out_shape)

        #print("hessians_out:", hessians_out)
        #print("grads_out:", grads_out)
        return hessians_out, grads_out

    def decomposed_parameters(self,pars):
        """Separate input parameters into 'interest' and 'nuisance' lists,
           keeping tracking of their original 'indices' w.r.t. catted format.
           Mainly used for decomposing Hessian matrix."""
        interest, fixed, nuisance = self.decomposed_parameter_shapes()
        interest_i = {} # indices of parameters in Hessian/stacked pars
        nuisance_i = {}
        interest_p = {} # parameters themselves
        nuisance_p = {} 
        i = 0
        #print("Decomposition order:")
        for ka,a in pars.items(): # Hessian should be ordered according to this parameter dictionary!
            interest_p[ka] = {}
            nuisance_p[ka] = {}
            interest_i[ka] = {}
            nuisance_i[ka] = {}
            for kp,p in a.items():
                if kp in nuisance[ka].keys():
                    N = c.prod(nuisance[ka][kp]) or 1
                    nuisance_p[ka][kp] = p
                    nuisance_i[ka][kp] = (i, N)
                    #print("   {0}::{1} (nuisance)".format(ka,kp))
                elif kp in interest[ka].keys():
                    N = c.prod(interest[ka][kp]) or 1
                    interest_p[ka][kp] = p
                    interest_i[ka][kp] = (i, N)
                    #print("   {0}::{1} (interest)".format(ka,kp))
                elif kp in fixed[ka].keys():
                    # Ignore the fixed parameters, they are bystanders in the Hessian calculation
                    N = 0
                else:
                    msg = "Tried to decompose parameters into 'interest' and 'nuisance' groups, however an unrecognised parameter was detected in the list for analysis {0}: par was {1}".format(ka, kp)
                    raise ValueError(msg)
                i+=N
        return interest_i, interest_p, nuisance_i, nuisance_p

    def sub_Hessian(self,H,pari,parj,idim=-1,jdim=-2):
        """Extract sub-Hessian matrix from full Hessian H,
           using dictionary that provides indices for selected
           parameters in H"""
        ilist = []
        for ai in pari.values():
            for i,Ni in ai.values():
                ilist += [ix for ix in range(i,i+Ni)] 
        jlist = []
        for aj in parj.values():
            for j,Nj in aj.values():
                jlist += [jx for jx in range(j,j+Nj)] 
        #print("H.shape:",H.shape)
        #print("ilist:",ilist)
        #print("jlist:",jlist)
        # Use gather to extract row/column slices from Hessian
        if len(ilist)>0:
            subH_i = tf.gather(H,      ilist, axis=idim)
            if len(jlist)>0:
                subH = tf.gather(subH_i, jlist, axis=jdim)
            else:
                subH = None
        else:
            subH = None
        return subH

    def sub_grad(self,grad,pari,idim=-1):
        """Extract sub-gradient vector from full grad,
           using dictionary that provides indices for selected
           parameters in grad"""
        ilist = []
        for ai in pari.values():
            for i,Ni in ai.values():
                ilist += [ix for ix in range(i,i+Ni)] 
        sub_grad = tf.gather(grad, ilist, axis=idim)
        return sub_grad

    def decompose_Hessian(self,H,parsi,parsj):
        """Decompose Hessian matrix into
           parameter blocks"""
        Hii = self.sub_Hessian(H,parsi,parsi)
        Hjj = self.sub_Hessian(H,parsj,parsj)
        Hij = self.sub_Hessian(H,parsi,parsj) #Off-diagonal block. Symmetric so we don't need both.
        return Hii, Hjj, Hij

    def quad_loglike_prep(self,samples):
        """Compute second-order Taylor expansion of log-likelihood surface
           around input parameter point(s), and compute quantities needed
           for analytic determination of profile likelihood for fixed signal
           parameters, under this approximation."""
        #print("Computing Hessian and various matrix operations for all samples...")
        H, g = self.Hessian(samples)
        #print("H:", H)
        #print("g:", g) # Should be close to zero if fits worked correctly
        pars = self.get_pars() # This is what Hessian uses internally
        #print("pars:", pars)
        #print("self.pars:", self.pars)
        interest_i, interest_p, nuisance_i, nuisance_p = self.decomposed_parameters(pars)
        #print("descaled_pars:", pars)
        #print("samples:", samples)
        #print("interest_p:", interest_p)
        #print("nuisance_p:", nuisance_p)
        #print("interest_i:", interest_i)
        #print("nuisance_i:", nuisance_i)
        Hii, Hnn, Hin = self.decompose_Hessian(H,interest_i,nuisance_i)

        #print("Hii:", Hii)
        #print("Hnn.shape:", Hnn.shape if Hnn is not None else None)
        #print("Hin.shape:", Hin.shape if Hin is not None else None)
        if Hnn is None: # Could be None if there aren't any nuisance parameters!
            A = None
            B = None
        else:
            Hnn_inv = tf.linalg.inv(Hnn)
            #print("Hii:", Hii)
            #print("Hin:", Hin)
            #print("Hnn:", Hnn)
            #print("Hnn_inv:", Hnn_inv)
            gn = self.sub_grad(g,nuisance_i)
            #print("gn:", gn)
            # Hmm, gn should always be zero if we maximised the logL w.r.t. the nuisance parameters at the expansion point? Should be at a maxima in that direction?
            #gn *= 0. # Test effect of enforcing this
            A = tf.linalg.matvec(Hnn_inv,gn)
            B = tf.linalg.matmul(Hnn_inv,Hin,transpose_b=True) # TODO: Not sure if transpose needed here. Doesn't seem to make a difference, which seems a little odd.
        #print("...done!")
        #print("A:", A)
        #print("B:", B)
        kwargs = {"A":A, "B":B, "interest":interest_p, "nuisance":nuisance_p}
        #print("in quad prep:")
        #print("kwargs:", kwargs)
        return kwargs

    def log_prob_quad_f(self,samples):
        """Return a function that can be used to compute the profile log-likelihood
           for fixed signal parameters, for many different signal hypotheses, using a 
           second-order Taylor expandion of the likelihood surface about a point to
           determine the profiled nuisance parameter values. 
           Should be used after pars are fitted to the desired expansion point, e.g.
           global best fit, or perhaps a null hypothesis point"""
        #print("quad_loglike_f; samples:", samples)
        prep_kwargs = self.quad_loglike_prep(samples)
        f = mm.tools.func_partial(self._log_prob_quad,samples=samples,**prep_kwargs)
        return f

    def nuisance_quad_f(self,samples):
        """Return a function that can be used to compute profiled (i.e. fitted, MLE) nuisance
           parameters for fixed signal parameters, for many different signal hypotheses, using a 
           second-order Taylor expandion of the likelihood surface about a point to
           determine the profiled nuisance parameter values. 
           Should be used after pars are fitted to the desired expansion point, e.g.
           global best fit, or perhaps a null hypothesis point"""
        prep_kwargs = self.quad_loglike_prep(samples)
        f = mm.tools.func_partial(self._nuisance_quad,**prep_kwargs)
        return f

    def _nuisance_quad(self,signal,A,B,interest,nuisance):
        """Compute nuisance parameter MLEs using pre-computed Taylor expansion
           parameters (for many samples) for a set of signal hypotheses"""

        # Make sure format of signal parameters matches the known "interest" parameters
        # for all analyses (also culls out any unnecessary parameters)
        parlist_init = {}
        for ka,a in interest.items():
            if ka not in signal.keys():
                raise ValueError("No test signals provided for analysis {0}".format(ka))
            parlist_init[ka] = {}
            for kp in a.keys():
                if kp not in signal[ka].keys():
                    raise ValueError("No test signals provided for parameter {0} in analysis {1}".format(kp,ka))
                #print("signal...", signal[ka][kp])
                parlist_init[ka][kp] = signal[ka][kp]

        # Make sure parameters are tensorflow variables:
        parlist = c.convert_to_TF_variables(parlist_init)

        #print("parlist:", parlist)
        #print("interest:", interest)

        if A is None or B is None:
            # No nuisance parameters exist for this analysis! So no expansion to be done. 
            theta_prof_dict = None
        else:
            # First we need to broadcast the "signal" batch shape against the expansion-point batch shape
            par_shapes = self.parameter_shapes()
            batch_shape_1 = c.all_dist_batch_shape(parlist,par_shapes)
            batch_shape_2 = c.all_dist_batch_shape(interest,par_shapes)
            batch_shape_3 = c.all_dist_batch_shape(nuisance,par_shapes)
            batch_shape_4 = c.get_bcast_shape(batch_shape_1,batch_shape_2)
            batch_shape = c.get_bcast_shape(batch_shape_3,batch_shape_4)
            #parlist_bcast = c.bcast_all_dist_batch_shape(parlist,par_shapes,batch_shape)
            #interest_bcast = c.bcast_all_dist_batch_shape(interest,par_shapes,batch_shape)
            #nuisance_bcast = c.bcast_all_dist_batch_shape(nuisance,par_shapes,batch_shape)

            flatten = False # If flattening then need to undo it to restore original signal batch shape
            s, s_bshape, s_names = c.cat_pars_to_tensor(parlist,par_shapes,flatten) 
            s_0, s0_bshape, s0_names = c.cat_pars_to_tensor(interest,par_shapes,flatten) # stacked interest parameter values at expansion point
            theta_0, t0_bshape, t0_names = c.cat_pars_to_tensor(nuisance,par_shapes,flatten) # stacked nuisance parameter values at expansion point

            # print("s_bshape:", s_bshape)
            # print("s0_bshape:", s0_bshape)
            # print("t0_bshape:", t0_bshape)
 
            # print("s.shape:", s.shape)
            # print("s_0.shape:", s_0.shape)
            # print("theta_0.shape:", theta_0.shape)
            # print("A.shape:", A.shape)
            # print("B.shape:", B.shape)

            # print("A:", A)
            # print("B:", B)

            # See test_jointdistribution (test_quad_prep) for explanation of how these shapes should work
            Ashift = theta_0 - A
            #sdiff_shape = c.get_bcast_shape(s.shape,s_0.shape) # This is redundant I should think...
            #sdiff = tf.broadcast_to(s,sdiff_shape) - tf.broadcast_to(s_0,sdiff_shape)
            sdiff = s - s_0
            # print("s:", s)
            # print("s_0:", s_0)
            # print("sdiff:", sdiff)
            # print("sdiff.shape:", sdiff.shape)
            Bvec = tf.linalg.matvec(B,sdiff)
            # print("Bvec:", Bvec)
            #theta_prof = Ashift - Bvec
            theta_prof = theta_0 - Bvec
            #batch_shape = theta_prof.shape[:-1] # Last dimension is flattened parameters, the rest can be thought of as the batch shape
            # ^^^ now have a more general method for restoring batch shape

            # de-stack analytically profiled nuisance parameters
            theta_prof_dict = c.decat_tensor_to_pars(theta_prof,nuisance,par_shapes,batch_shape) 

            # print("theta (in):", theta_0)
            # print("theta_prof:", theta_prof)
            # print("theta_prof_dict:", theta_prof_dict)

        return theta_prof_dict
 
    def _log_prob_quad(self,signal,samples,**kwargs):
        """Compute loglikelihood using pre-computed Taylor expansion
           parameters (for many samples) for a set of signal hypotheses"""

        # Get the profiled nuisance parameters under the Taylor expansion.
        theta_prof_dict = self._nuisance_quad(signal,**kwargs)

        if theta_prof_dict is None:
            # No nuisance parameters exist for this analysis! So no expansion to be done. Just evaluate the signal directly.
            # Note: there are some shape issues, though. When we use parameters that have been fitted to samples, those
            # fitted parameters have an extra 0 dimension, i.e. the sample dimension (one parameter for each sample). This
            # is missing when there are no nuisance parameters, so we need to add it.
            #expanded_pars = c.deep_expand_dims(c.convert_to_TF_variables(signal),axis=0)
            expanded_pars = c.convert_to_TF_constants(signal)
            # Compute -2*log_prob
            # print("expanded_pars:", expanded_pars)
            joint = JointDistribution(self.analyses.values(),expanded_pars)
        else:
            joint = JointDistribution(self.analyses.values(),c.deep_merge(signal,theta_prof_dict))

        batch_shape = self.bcast_batch_shape_tensor()
        # print("in _log_prob_quad: batch_shape = ",batch_shape)

        # Need to match samples to the batch shape (i.e. broadcast over the 'hypothesis' dimension)
        # This is a little confusing, but basically need to make the sample_shape+batch_shape for the sample
        # match the batch_shape of the JointDistribution. Will assume axis 0 is always the "number of samples", so extra dims
        # are to be inserted into the batch dims of the sample at axis 1.
        consistent, batch_shape = c.deep_equals(joint.batch_shape_tensor())
        if not consistent:
            msg = "Inconsistent batch dimensions found! Batch shape dictionary was: {0}".format(joint.batch_shape_tensor())
            raise ValueError(msg)
        s_batch_shape = joint.sample_batch_shape(samples)
        if s_batch_shape==() and (theta_prof_dict is None) : s_batch_shape = [0] # Interpret as one batch dim when zero. This is a little hacky, I probably need to tighten up the shape propagation.
        n_new_dims = len(batch_shape) - len(s_batch_shape)
        matched_samples = samples
        for i in range(n_new_dims):
            matched_samples = c.deep_expand_dims(matched_samples,axis=1)
        log_prob = joint.log_prob(matched_samples)
        # print("batch_shape:", batch_shape)
        # print("s_batch_shape:", s_batch_shape)
        # print("n_new_dims:", n_new_dims)
        # print("samples:", samples)
        # print("matched_samples:", matched_samples)
        # print("log_prob:", log_prob)
        return log_prob #c.squeeze_to(q,2,dont_squeeze=[0])

    def bcast_batch_shape_tensor(self):
        """The built-in batch_shape_tensor method for NamedJointDistribution in
           tensorflow_probability returns a dictionary of batch shapes, one for
           each component of the JointDistribution.
           Here, we instead return a *single* batch_shape_tensor, representing
           a common consistent batch_shape for the whole JointDistribution. It
           will be an error if no consistent such shape exists after broadcasting
           rules are applied.

           TODO: doesn't return a tensor. Should change the name and all references
           to reflect this, i.e. it will only work with eager evaluation. That is
           true for a lot of jmctf probably.
        """
        
        all_batch_shapes = self.batch_shape_tensor()
        #print("all_batch_shapes:", all_batch_shapes)

        # First pass: find the batch shape with the most dimensions. All shapes
        # will be broadcast to this number of dims.
        ndims = 0
        for d,shape in all_batch_shapes.items():
            #print("d: {0}, shape: {1}, shape.shape[0]: {2}".format(d,shape,shape.shape[0]))
            if shape.shape[0] > ndims: ndims = shape.shape[0]
        
        # Second pass: attempt to broadcast everything to ndims
        out_shape = tuple(1 for i in range(ndims))
        #print("out_shape:", out_shape)
        for d,shape in all_batch_shapes.items():
             msg = "Could not obtain consistent batch_shape across all components of the JointDistribution! Please check that all your distribution parameters will result in distribution components whose batch_shape dimensions can be broadcast against each other. Failure occurred when broadcasting distribution named '{0}' with shape {1}, to shape {2}".format(d,shape,out_shape)
             try:
                 out_shape = c.get_bcast_shape(shape,out_shape)
             except ValueError as e:
                 raise ValueError(msg) from e
             if -1 in out_shape:
                 raise ValueError(msg + ": -1 was detected in shape!")

        return out_shape

    def parameter_shapes(self):
        """Returns a dictionary describing the shapes of all input parameters for each analysis 
           required to produce a single event of the basic event shape for all component distributions. 
           
           Note: There is a similar function built-in, "param_shape", but it doesn't really make 
           sense for JointDistribution since it only accepts one input 'event_shape', whilst this 
           can obviously be different across the various component distributions. This function
           also works for analysis parameters rather than distribution parameters."""
        param_shapes = {name: a.parameter_shapes() for name,a in self.analyses.items()}
        return param_shapes

    def decomposed_parameter_shapes(self):
        """Returns three dictionaries whose structure explains how parameters should be supplied
           to this object"""
        interest  = {a.name: a.interest_parameter_shapes() for a in self.analyses.values()}
        fixed = {a.name: a.fixed_parameter_shapes() for a in self.analyses.values()}
        nuis  = {a.name: a.nuisance_parameter_shapes() for a in self.analyses.values()} 
        return interest, fixed, nuis

    def event_shapes(self):
        """Returns dictionary explaining the 'base' event shapes for each distribution in each analysis.
           Analysis names are added as prefixes to the dict keys to match conventions for Osamples and
           Asamples members (generated in constructor).

           Result is equivalent to calling self.event_shape_tensor(), however
           event_shape_tensor() is a method inherited from 
           JointDistributionNamed that cannot be called if self.has_pars() is
           False. In contrast this method will always work.
        """
        all_event_shapes = {}
        for a in self.analyses.values():
            all_event_shapes.update(c.add_prefix(a.name,a.event_shapes())) 
        return all_event_shapes

    def sample_batch_shape(self, samples):
        """Returns a shape tuple describing the shape of the dimensions of 
           'sample' that would be interpreted as sample+batch dimensions if 'sample'
           is input into methods of this object such as log_prob"""
        event_shape = self.event_shapes()
        s_batch_shape = c.sample_batch_shape(samples, event_shape)
        return s_batch_shape

    def expected_batch_shape_nuis(self, par_shapes, samples=None, sample_shape=None):
        """Returns a shape tuple describing the batch dimensions of the
           JointDistribution that would be obtained by fitting the nuisance
           parameter of the current JointDistribution to 'samples' using
           fixed 'parameters'.

           Can provide either the actual samples that would be fitted, or else just
           their sample+batch shape.
        """
        event_shapes = self.event_shapes() 
        par_shapes = self.parameter_shapes()
        dist_batch_shape = c.all_dist_batch_shape(parameters, par_shapes)
        if sample_shape is not None and samples is not None:
            raise ValueError("Please provide only one of 'samples' or 'sample_shape' as arguments")
        elif sample_shape is None and samples is not None:
            bcast_samples = c.bcast_sample_batch_shape(samples, event_shapes, dist_batch_shape)
            bcast_dist_batch_shape = c.sample_batch_shape(bcast_samples, event_shapes)
        elif samples is None and sample_shape is not None:
            # TODO: Write broadcasting functions that can just work with the shapes!
            raise Exception("Not implemented!")
        else:
            raise ValueError("Either 'samples' or 'sample_shape' must be provided!")
        print("in expected_batch_shape_nuis:")
        print("  event_shapes:", event_shapes)
        print("  par_shapes:", par_shapes)
        print("  dist_batch_shape:", dist_batch_shape)
        print("  bcast_dist_batch_shape:", bcast_dist_batch_shape)
        return bcast_dist_batch_shape
