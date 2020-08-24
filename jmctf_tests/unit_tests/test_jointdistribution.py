"""Unit tests for JointDistribution class"""

import pytest
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import jmctf.common as c
from jmctf import JointDistribution
from jmctf_tests.analysis_class_register import get_id_list, get_obj, get_test_hypothesis, get_hypothesis_lists

# Number of samples to draw during tests
N = 5 # Needs to match number of hypotheses for Hessian calculation (in reality would use parameters fit to each sample)

# Hessian test shape outputs
Hessians_shape = {
        "NormalAnalysis": (N, 1, 1) ,
        "NormalTEAnalysis": (N, 2, 2) ,
        "BinnedAnalysis": (N, 4, 4)
        }

# @pytest.fixture(scope="module"
#                ,params=[(get_obj(name), get_test_hypothesis(name), H) for name,H in Hessians_shape.items()]
#                       +[(get_obj(name), get_hypothesis_lists(name), H) for name,H in Hessians_shape.items()]
#                ,ids = [name + " (single hypothesis)" for name in Hessians_shape.keys()]
#                      +[name + " (multiple hypotheses)" for name in Hessians_shape.keys()]
#                 )
@pytest.fixture(scope="module"
               ,params=[(get_obj(name), get_hypothesis_lists(name), H) for name,H in Hessians_shape.items()]
               ,ids = [name + " (multiple hypotheses)" for name in Hessians_shape.keys()]
                )
def analysis_params_Hshape(request):
    analysis, params, H_shape = request.param
    return analysis, params, H_shape 

@pytest.fixture(scope="module")
def analysis(analysis_params_Hshape):
    analysis, params, H_shape = analysis_params_Hshape
    return analysis 

@pytest.fixture(scope="module")
def pars(analysis_params_Hshape):
    analysis, params, H_shape = analysis_params_Hshape
    pars = {analysis.name: params}
    return pars

@pytest.fixture(scope="module")
def Hshape(analysis_params_Hshape):
    analysis, params, H_shape = analysis_params_Hshape
    return H_shape 

@pytest.fixture(scope="module")
def joint(analysis,pars):
    """Construct joint distribution object for testing
       This version uses only the 0th hypothesis.
    """
    print("pars:", pars)
    print("pars[0]:", c.extract_ith(pars,0,keep_axis=True))
    jd = JointDistribution([analysis],c.extract_ith(pars,0,keep_axis=True))
    print("batch shape:", jd.batch_shape_tensor())
    return jd

@pytest.fixture(scope="module")
def samples(joint):
    x = joint.sample(N)
    print("x:", x) 
    return x 

@pytest.fixture(scope="module")
def joint_fitted_nuisance_log_prob_pars(joint,samples,pars):
    """Obtain joint distribution with nuisance parameters
       fitted to samples"""
    pars0 = c.extract_ith(pars,0,keep_axis=True) # The fixed non-nuisance parameters to use in the fits
    log_prob, joint_fitted_nuisance, fitted_pars_nuisance = joint.fit_nuisance(samples, pars0, log_tag='test_jointdistribution')
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
def internal_pars(joint):
    """Retrieve internal parameters of joint distribution"""
    return joint.get_pars()

@pytest.fixture(scope="module")
def decomposed_parameters(joint,internal_pars):
    interest_i, interest_p, nuisance_i, nuisance_p = joint.decomposed_parameters(internal_pars)
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

@pytest.fixture(scope="module")
def sub_hessian_shapes(hessian,decomposed_parameters,decomposed_hessian,Hshape):
    H, g = hessian
    Hii, Hnn, Hin = decomposed_hessian
    interest_i, interest_p, nuisance_i, nuisance_p = decomposed_parameters
    # Compute number of slots in Hessian that each parameter should occupy
    # Assumes that parameters were correctly decomposed (should have a separate test for this)
    islots = 0
    for pars in interest_i.values():
        for (i,n) in pars.values():
            islots += n
    nslots = 0
    for pars in nuisance_i.values():
        for (i,n) in pars.values():
            nslots += n
    if islots>0 and nslots>0:
        Hii_shape = (Hshape[0],islots,islots)
        Hnn_shape = (Hshape[0],nslots,nslots)
        Hin_shape = (Hshape[0],islots,nslots)
    elif nslots>0:
        Hii_shape = None
        Hnn_shape = H.shape #(Hshape[0],nslots,nslots)
        Hin_shape = None
    elif islots>0:
        Hii_shape = H.shape #(Hshape[0],islots,islots)
        Hnn_shape = None
        Hin_shape = None
    else:
        # No parameters of any kind?
        Hii_shape = None
        Hnn_shape = None
        Hin_shape = None
    return Hii_shape, Hnn_shape, Hin_shape
  
def test_fitted_log_prob(joint_fitted_nuisance,fitted_log_prob,samples):
    recalculated_log_prob = joint_fitted_nuisance.log_prob(samples)
    print("fitted_log_prob:", fitted_log_prob)
    print("recalculated_log_prob:", recalculated_log_prob)
    print("joint_fitted_nuisance.get_pars():", joint_fitted_nuisance.get_pars())
    assert c.tf_all_equal(fitted_log_prob, recalculated_log_prob)
        
def test_hessian_shape(hessian,Hshape):
    H, g = hessian
    print("H:", H)
    print("Hshape:", Hshape)
    Hshape1 = list(Hshape)
    Hshape1[0] = 1 # Case for no nuisance parameters, so no nuisance fits, so only single Hessian returns.
    assert H.shape == Hshape or H.shape == Hshape1

def test_sub_hessian_shapes(decomposed_hessian,sub_hessian_shapes):
    Hii, Hnn, Hin = decomposed_hessian
    Hii_shape, Hnn_shape, Hin_shape = sub_hessian_shapes
    print("Hii:", Hii)
    print("Hnn:", Hnn)
    print("Hin:", Hin)
    if Hii_shape is None: assert Hii == None
    else: assert Hii.shape==Hii_shape
    if Hnn_shape is None: assert Hnn == None
    else: assert Hnn.shape==Hnn_shape
    if Hin_shape is None: assert Hin == None
    else: assert Hin.shape==Hin_shape

def test_quad_prep(quad_prep,samples,joint):
    A = quad_prep["A"]
    B = quad_prep["B"]
    interest = quad_prep["interest"]
    nuisance = quad_prep["nuisance"]

    # Check if there are any nuisance parameters
    count_nuisances = 0
    for analysis in nuisance.values():
        for parameter in analysis.values():
            count_nuisances += 1

    if count_nuisances==0:
        assert A is None
        assert B is None
    else:
        par_shapes = joint.parameter_shapes()
        s_0, s_batch_shape, s_col_names = c.cat_pars_to_tensor(interest,par_shapes) # stacked interest parameter values at expansion point(s)
        theta_0, t_batch_shape, theta_col_names = c.cat_pars_to_tensor(nuisance,par_shapes) # stacked nuisance parameter values at expansion point(s)
        print("theta_0.shape:", theta_0.shape)
        (n, p) = theta_0.shape
        # Shape requirements:
        # N = number of samples
        # p = number of (scalarized) nuisance parameters
        assert n == N
        assert A.shape == (N,p)
        assert B.shape == (N,p,p)

def test_fitted_pars(joint_fitted_nuisance,fitted_pars):
    """Check that parameters returned from fit match those in the accompanying "fitted" distribution"""
    print("fitted_pars['all']:", fitted_pars['all'])
    print("joint_fitted_nuisance.get_pars():",joint_fitted_nuisance.get_pars()) 
    assert c.deep_all_equal(fitted_pars['all'], joint_fitted_nuisance.get_pars())

def test_quad_logl_nuisance(joint_fitted_nuisance,fitted_pars,quad_prep,samples):
    """Check that quad-estimate nuisance parameters match fitted ones at the expansion point"""
    signal = fitted_pars["fixed"]
    quad_nuis_pars = joint_fitted_nuisance._nuisance_quad(signal,samples,**quad_prep)
    nuis_pars = fitted_pars["fitted"]
    print("signal:", signal)
    print("quad_nuis_pars:", quad_nuis_pars)
    print("nuis_pars:", nuis_pars)
    if quad_prep["A"] is None:
        assert quad_nuis_pars is None
    else:
        assert c.deep_all_equal(quad_nuis_pars,nuis_pars)

def test_quad_logl(joint_fitted_nuisance,fitted_log_prob,fitted_pars,quad_prep,samples):
    """Check that shapes for log_prob_quad calculation make sense, and that values match
       the true log_prob at the expansion point"""
    #A, B, interest, nuisance = quad_prep
    #q = joint_fitted_nuisance.neg2loglike_quad(internal_pars,A,B,interest,nuisance,samples)

    # First make sure we understand the fitted parameters in the joint distribution
    # Obtain the quadratic estimate of the log_prob
    f = joint_fitted_nuisance.log_prob_quad_f(samples)
    log_prob_quad = f(fitted_pars['fixed']) # the 'fixed' parameters include the 'signal' ones
    log_prob = fitted_log_prob

    # Test shapes. 
    print("samples:", samples)
    print("fitted_pars['fixed']:", fitted_pars['fixed'])
    print("log_prob_quad.shape:",             log_prob_quad.shape)
    print("c.deep_size(samples):",            c.deep_size(samples))
    print("c.deep_size(fitted_pars['fixed']):", c.deep_size(fitted_pars['fixed']))

    assert log_prob_quad.shape[0]==c.deep_size(samples)
    assert log_prob_quad.shape[1]==c.deep_size(fitted_pars['fixed'])

    # Test values; quad log_prob should equal true log_prob at the expansion points. And we
    # have only calculated it at the expansion points, so all values should be equal
    print("log_prob_quad:", log_prob_quad)
    print("fitted_log_prob:", fitted_log_prob)
    assert c.tf_all_equal(log_prob_quad, fitted_log_prob)

# @pytest.mark.parametrize(
#     "analysis,single_hypothesis,hypothesis_list,Hessian_list",
#     [(get_obj(name), get_test_hypothesis(name), get_hypothesis_lists(name), H_shape) for name,H_shape in Hessians_shape_list.items()],
#     ids = Hessians_shape_list.keys()
#     )
# def test_Hessian_shape_single_with_list(analysis,single_hypothesis,hypothesis_list,Hessian_list):
#     """Compute Hessian for list of parameters, for samples generated
#        from a single hypothesis"""
#     in_pars = {analysis.name: single_hypothesis}
#     joint = JointDistribution([analysis],in_pars)
#     x = joint.sample(4)
#     print("x:", x)
#     test_pars = {analysis.name: hypothesis_list}
#     print("test_pars:", test_pars)
#     H = joint.Hessian(test_pars,x)
#     print("H:", H)
#     #assert False

