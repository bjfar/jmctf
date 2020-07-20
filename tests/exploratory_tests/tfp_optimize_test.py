"""TensorFlow probability comes with some optimization tools
   built-in. These are just exploratory tests to see if I can
   use them to replace my custom-written massminimize tool."""

import numpy as np
import functools
import contextlib
import time
import tensorflow as tf
import tensorflow_probability as tfp
from jmctf import NormalAnalysis, BinnedAnalysis, JointDistribution

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
# ==================


# Will use the same JointDistribution as in the quickstart examples.

# make_norm
sigma = 2.
norm = NormalAnalysis("Test normal", 5, sigma)
# make_binned
# (name, n, b, sigma_b)
bins = [("SR1", 10, 9, 2),
        ("SR2", 50, 55, 4)]
binned = BinnedAnalysis("Test binned", bins)
# make_joint
joint = JointDistribution([norm,binned])

# JointDistribution here is a child class of the tfp class 
# JointDistributionNamed, so it should work pretty well with
# other tfp tools.

np.random.seed(12345)

dim = 100
batches = 500
minimum = np.random.randn(batches, dim)
scales = np.exp(np.random.randn(batches, dim))

@make_val_and_grad_fn
def quadratic(x):
  return tf.reduce_sum(input_tensor=scales * (x - minimum)**2, axis=-1)

@make_val_and_grad_fn
def neg2logL(pars):

  return tf.reduce_sum(input_tensor=scales * (x - minimum)**2, axis=-1)

def neg2LogL(pars,const_pars,analyses,data,transform=None):
    """General -2logL function to optimise
       TODO: parameter 'transform' feature not currently in use, probably doesn't work correctly
    """
    #print("In neg2LogL:")
    #print("pars:", c.print_with_id(pars,id_only))
    #print("const_pars:", c.print_with_id(const_pars,id_only))
    if transform is not None:
        pars_t = transform(pars)
    else:
        pars_t = pars
    if const_pars is None:
        all_pars = pars_t
    else:
        all_pars = c.deep_merge(const_pars,pars_t)

    # Sanity check: make sure parameters haven't become nan somehow
    anynan = False
    nanpar = ""
    for a,par_dict in pars.items():
        for p, val in par_dict.items():
            if tf.math.reduce_any(tf.math.is_nan(val)):
                anynan = True
                nanpar += "\n    {0}::{1}".format(a,p)
    if anynan:
        msg = "NaNs detected in parameter arrays during optimization! The fit may have become unstable and wandering into an invalid region of parameter space; please check your analysis setup. Parameter arrays containing NaNs were:{0}".format(nanpar)
        raise ValueError(msg)

    # Parameters will enter this function pre-scaled such that MLEs have variance ~1
    # So we need to set the pre-scaled flag for the JointDistribution constructor to
    # avoid applying the scaling a second time.
    joint = JointDistribution(analyses.values(),all_pars,pre_scaled_pars=True)
    q = -2*joint.log_prob(data)
    #print("q:", q)
    #print("all_pars:", all_pars)
    #print("logL parts:", joint.log_prob_parts(data))

    if tf.math.reduce_any(tf.math.is_nan(q)):
        # Attempt to locate components generating the nans
        component_logprobs = joint.log_prob_parts(data)
        nan_components = ""
        for comp,val in component_logprobs.items():
            if tf.math.reduce_any(tf.math.is_nan(val)):
                nan_components += "\n    {0}".format(comp)                
        msg = "NaNs detect in result of neg2LogL calculation! Please check that your input parameters are valid for the distributions you are investigating, and that the fit is stable! Components of the joint distribution whose log_prob contained nans were:" + nan_components
        raise ValueError(msg)
    total_loss = tf.math.reduce_sum(q)
    return total_loss, q, None, None

# Make all starting points (1, 1, ..., 1). Note not all starting points need
# to be the same.
start = tf.ones((batches, dim), dtype='float64')

@tf.function
def batch_multiple_functions():
  return tfp.optimizer.lbfgs_minimize(
      quadratic, initial_position=start,
      stopping_condition=tfp.optimizer.converged_all,
      max_iterations=100,
      tolerance=1e-8)

results = run(batch_multiple_functions)
print('All converged:', np.all(results.converged))
print('Largest error:', np.max(results.position - minimum))

