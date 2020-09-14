"""Some specialised unit tests for the 'quad' log_prob functions of JointDistritubiton,
   including some plotting for manual inspection that log_prob curves look correct.
   
   This is mainly testing the 'quad' machinery rather than anything analysis-specific,
   so it only runs on some manually created test cases.
   """

import pytest
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import jmctf.common as c
from jmctf import JointDistribution
from jmctf_tests.analysis_class_register import get_id_list, get_obj, get_test_hypothesis, get_hypothesis_lists, get_hypothesis_curves

# Common fixtures for common test setup tasks
from jmctf_tests.common_fixtures import analysis, pars, pars0, joint0, samples, joint_fitted_nuisance_log_prob_pars, joint_fitted_nuisance, fitted_log_prob, fitted_pars, hessian, internal_pars,  decomposed_parameters, decomposed_hessian, quad_prep, quad_f


has_surface = ["NormalTEAnalysis"]

# Number of samples to draw during tests
N = 2 # Doing plotting, so don't want tonnes of curves confusing things
@pytest.fixture(scope="module")
def Nsamples():
    return N

pars_id = [(get_obj(name), curve, name+"_"+ID) for name in has_surface for ID,curve in get_hypothesis_curves(name).items()]
params = [x[0:2] for x in pars_id]
ids = [x[2] for x in pars_id]

# "Entry point" fixture providing the analysis objects and parameters for them.
@pytest.fixture(scope="module",params=params,ids=ids)
def analysis_params(request):
    analysis, params = request.param
    return analysis, params

def test_plot_quad_logl(analysis,pars,samples):
    """Create plots of test logl curves, comparing direct fits with 'quad' versions"""
    print("pars:", pars)
    print("samples:", samples)
  
    # We want to fit all samples for all parameters, so we need to make sure the batch shapes
    # can be broadcast against each other. Easiest way is to insert some extra dimensions into
    # both (since they are both 1D batches to start with)
    pars_batch = c.deep_expand_dims(pars,axis=0)
    samples_batch = c.deep_expand_dims(samples,axis=1)

    joint = JointDistribution([analysis],pars_batch)
    log_prob, joint_fitted_nuisance, fitted_pars_nuisance = joint.fit_nuisance(samples_batch)

    print("log_prob:", log_prob)
    print("fitted_pars_nuisance['fitted']:", fitted_pars_nuisance['fitted'])
    print("fitted_pars_nuisance['fixed']:", fitted_pars_nuisance['fixed'])
    print("pars_batch:", pars_batch)

    # The 'quad' log_prob, expanding about every nuisance BF point for every hypothesis (i.e. NOT what is done in lee-correction)
    f = joint_fitted_nuisance.log_prob_quad_f(samples_batch)
    log_prob_quad = f(pars_batch) #fitted_pars_nuisance['fixed']) # the 'fixed' parameters include the 'signal' ones (EDIT: can just use pars_batch, same thing)

    # The 'quad' log_prob, expanding just once about the global BF point amongst input hypotheses, per sample (i.e. what IS done in lee-correction)
    
    #joint_fitted_nuisance


    print("log_prob:", log_prob)
    print("log_prob_quad (re-fit):", log_prob_quad)
    #print("log_prob_quad (global BF expansion):"
    assert False
