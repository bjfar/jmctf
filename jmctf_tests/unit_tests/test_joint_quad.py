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

import matplotlib.pyplot as plt

# Common fixtures for common test setup tasks
from jmctf_tests.common_fixtures import analysis, pars, pars0, joint0, samples, joint_fitted_nuisance_log_prob_pars, joint_fitted_nuisance, fitted_log_prob, fitted_pars, hessian, internal_pars,  decomposed_parameters, decomposed_hessian, quad_prep, quad_f


has_curve = ["NormalAnalysis","NormalTEAnalysis","BinnedAnalysis_single","BinnedAnalysis"]
#has_curve = ["BinnedAnalysis_single"]

# Number of samples to draw during tests
N = 2 # Doing plotting, so don't want tonnes of curves confusing things
@pytest.fixture(scope="module")
def Nsamples():
    return N

pars_id = [(get_obj(name), ID, curve, "{0}_{1}".format(name,ID)) for name in has_curve for ID,curve in get_hypothesis_curves(name).items()]
params = [x[0:3] for x in pars_id] # analysis and curve data
ids = [x[3] for x in pars_id]

# "Entry point" fixture providing the analysis objects and parameters for them.
@pytest.fixture(scope="module",params=params,ids=ids)
def analysis_ID_params(request):
    analysis, IDpar, params = request.param
    return analysis, IDpar, params

@pytest.fixture(scope="module")
def analysis_params(analysis_ID_params):
    analysis, IDpar, params = analysis_ID_params
    return analysis, params

# The parameter that varies for the test curve
@pytest.fixture(scope="module")
def curve_par(analysis_ID_params):
    analysis, IDpar, params = analysis_ID_params
    return IDpar

# The name of the current test
@pytest.fixture
def test_name(request):
    return request.node.name

def test_plot_quad_logl(analysis,pars,samples,curve_par,test_name):
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
    print("pars_batch:", pars_batch)
    print("fitted_pars_nuisance['fitted']:", fitted_pars_nuisance['fitted'])
    print("fitted_pars_nuisance['fixed']:", fitted_pars_nuisance['fixed'])
    print("pars_batch:", pars_batch)

    # The 'quad' log_prob, expanding about every nuisance BF point for every hypothesis (i.e. NOT what is done in lee-correction)
    # So the following happens:
    # 1. (2,20) shape nuisance parameter fits obtained (2 samples * 20 hypotheses, broadcast against each other)
    #    These become the expansion points in the log_prob_quad evaluation
    # 2. (2,1) shaped samples are provided to create the log_prob_quad_f function
    # 3. (1,20) parameters are provided for log-likelihood evaluation
    #    These are the same parameters used as input to the fits, so should cause evaluation to occur exactly at the expansion points
    # 4. Result has shape (2,20)
    f = joint_fitted_nuisance.log_prob_quad_f(samples_batch)
    log_prob_quad = f(pars_batch) #fitted_pars_nuisance['fixed']) # the 'fixed' parameters include the 'signal' ones (EDIT: can just use pars_batch, same thing)


    # The 'quad' log_prob, expanding just once about the global BF point amongst input hypotheses, per sample (i.e. what IS done in lee-correction, more or less. Actually we use a null-hypothesis point rather than the BF, but the point is there is just one expansion point per sample)
    # So the following happens:
    # 1. (2,1) shape nuisance parameter fits obtained (2 samples * 1 hypothesis)
    #    These become the expansion points in the log_prob_quad evaluation
    # 2. (2,1) shaped samples are provided to create the log_prob_quad_f function
    # 3. (1,20) parameters are provided for log-likelihood evaluation
    #    These are DIFFERENT parameters to those used input to the fits, so should cause more non-trivial evaluation of the log_prob_quad
    #    function to occur. Results will be less accurate of course, but this is the "real-world" use case. Will make plots to check accuracy.
    log_prob_g, joint_fitted_all, fitted_pars_all = joint.fit_all(samples_batch)
    print("log_prob_g:", log_prob_g)
    print("fitted_pars_all['fitted']:", fitted_pars_all['fitted'])
    print("fitted_pars_all['fixed']:", fitted_pars_all['fixed'])

    f2 = joint_fitted_all.log_prob_quad_f(samples_batch)
    log_prob_quad_2 = f2(pars_batch)

    print("log_prob:", log_prob)
    print("log_prob_quad   (expanded from exact signal points):", log_prob_quad)
    print("log_prob_quad_2 (global BF expansion):", log_prob_quad_2)

    # Ok let's make some plots!

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Plot curve for each sample (0th axis of batch)
    if isinstance(curve_par, str):
        cpar, index = (curve_par, None)
    else:
        try: 
            cpar, index = curve_par
        except ValueError as e:
            msg = "Failed to interpret curve 'parameter' specification! Needs to be either a string, or a (string,index) tuple indicating which parameter (and which index if multivariate) is the one that varies for this test!"
            raise ValueError(msg) from e
 
    if index is None:
        x = pars[analysis.name][cpar]
    else:
        x = pars[analysis.name][cpar][:,index]

    first = True
    for y, y_quad_1, y_quad_2 in zip(log_prob,log_prob_quad,log_prob_quad_2):
        if first:
            ax.plot(x,y,c='k',label="Full numerical profiling")
            ax.plot(x,y_quad_1,c='g',ls='--',label="\"quad\" expansion at profiled points (i.e. no real expansion done)")
            ax.plot(x,y_quad_2,c='r',ls='--',label="\"quad\" expansion around single global best fit per sample")
            first = False
        else:
            # No labels this time
            ax.plot(x,y,c='k')
            ax.plot(x,y_quad_1,c='g',ls='--')
            ax.plot(x,y_quad_2,c='r',ls='--')
 
    ax.set_ylabel("log_prob")
    ax.set_xlabel(curve_par)
    ax.set_title("log_prob_quad curve test for analysis {0}, parameter {1}".format(analysis.name, curve_par))
    ax.legend(loc=0, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
    plt.tight_layout()
    fig.savefig("unit_test_output/log_prob_quad_comparison_{0}.png".format(test_name))

    assert False
