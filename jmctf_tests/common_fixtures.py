"""Stuff common to all tests in this directory. Import these fixtures into tests as
   required.
   Note: I thought conftest.py could be used for this, but I cannot get it to work
   when the "imported" fixture enters in the middle of the dependency chain, i.e.
   when it depends on things defined in the module
   
   NOTE: Cannot do "from common import *", dependencies don't get resolved properly.
   So here is a replacement "import *" to copy/paste:
   from jmctf_tests.common_fixtures import analysis, pars, pars0, joint0, samples, joint_fitted_nuisance_log_prob_pars, joint_fitted_nuisance, fitted_log_prob, fitted_pars, hessian, internal_pars,  decomposed_parameters, decomposed_hessian, quad_prep, quad_f
"""

import pytest
import jmctf.common as c
from jmctf import JointDistribution

@pytest.fixture(scope="module")
def square(x):
    return x**2

@pytest.fixture(scope="module")
def analysis(analysis_params):
    analysis, params = analysis_params
    return analysis 

@pytest.fixture(scope="module")
def pars(analysis_params):
    analysis, params = analysis_params
    pars = {analysis.name: params}
    return pars

@pytest.fixture(scope="module")
def pars0(analysis,pars):
    pars0 = c.extract_ith(pars,0) # The fixed non-nuisance parameters to use in the fits
    return pars0
 
@pytest.fixture(scope="module")
def joint0(analysis,pars0):
    """Construct joint distribution object for testing
       This version uses only the 0 hypothesis.
    """
    print("pars0:", pars0)
    jd = JointDistribution([analysis],pars0) 
    print("batch shape:", jd.batch_shape_tensor())
    return jd

@pytest.fixture(scope="module")
def samples(joint0,Nsamples):
    x = joint0.sample(Nsamples)
    print("x:", x) 
    return x 

@pytest.fixture(scope="module")
def joint_fitted_nuisance_log_prob_pars(joint0,samples,pars0):
    """Obtain joint distribution with nuisance parameters
       fitted to samples"""
    log_prob, joint_fitted_nuisance, fitted_pars_nuisance = joint0.fit_nuisance(samples, pars0, log_tag='test_jointdistribution')
    return log_prob, joint_fitted_nuisance, fitted_pars_nuisance

@pytest.fixture(scope="module")
def joint_fitted_nuisance(joint_fitted_nuisance_log_prob_pars):
    log_prob, joint_fitted_nuisance, fitted_pars_nuisance = joint_fitted_nuisance_log_prob_pars
    return joint_fitted_nuisance

@pytest.fixture(scope="module")
def fitted_log_prob(joint_fitted_nuisance_log_prob_pars):
    log_prob, joint_fitted_nuisance, fitted_pars_nuisance = joint_fitted_nuisance_log_prob_pars
    return log_prob 

@pytest.fixture(scope="module")
def fitted_pars(joint_fitted_nuisance_log_prob_pars):
    log_prob, joint_fitted_nuisance, fitted_pars_nuisance = joint_fitted_nuisance_log_prob_pars
    return fitted_pars_nuisance 

@pytest.fixture(scope="module")
def hessian(joint_fitted_nuisance,samples):
    """Compute Hessian around nuisance parameter
       points fitted to samples."""
    H, g = joint_fitted_nuisance.Hessian(samples)
    return H, g

@pytest.fixture(scope="module")
def internal_pars(joint0):
    """Retrieve internal parameters of joint distribution"""
    return joint0.get_pars()

@pytest.fixture(scope="module")
def decomposed_parameters(joint0,internal_pars):
    interest_i, interest_p, nuisance_i, nuisance_p = joint0.decomposed_parameters(internal_pars)
    #print("self.pars:", self.pars)
    print("internal_pars:", pars)
    print("samples:", samples)
    print("interest_p:", interest_p)
    print("nuisance_p:", nuisance_p)
    return interest_i, interest_p, nuisance_i, nuisance_p 

@pytest.fixture(scope="module")
def decomposed_hessian(joint_fitted_nuisance,hessian,decomposed_parameters):
    H, g = hessian
    interest_i, interest_p, nuisance_i, nuisance_p = decomposed_parameters
    print("interest_i:", interest_i) # i for indices (e.g. in Hessian)
    print("nuisance_i:", nuisance_i)
    print("interest_p:", interest_p)
    print("nuisance_p:", nuisance_p)
    Hii, Hnn, Hin = joint_fitted_nuisance.decompose_Hessian(H,interest_i,nuisance_i)
    return Hii, Hnn, Hin 

@pytest.fixture(scope="module")
def quad_prep(joint_fitted_nuisance,samples):
    prep_dict = joint_fitted_nuisance.quad_loglike_prep(samples)
    return prep_dict

@pytest.fixture(scope="module")
def quad_f(joint_fitted_nuisance,samples):
    return joint_fitted_nuisance.quad_loglike_f(samples)



