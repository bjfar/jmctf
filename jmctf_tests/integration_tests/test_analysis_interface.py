"""Integration tests that apply to all Analysis classes,
   and check that they conform correctly to the
   required interface"""

import pytest
import numpy as np
from tensorflow_probability import distributions as tfd
from jmctf import JointDistribution
from jmctf_tests.analysis_class_register import get_id_list, get_obj, get_test_hypothesis, get_hypothesis_lists

# Fixture to create analysis objects and provide test parameters to use with them 
@pytest.fixture(scope="module", 
                params=list(zip(get_obj().values(),get_test_hypothesis().values()))
                      +list(zip(get_obj().values(),get_hypothesis_lists().values())),
                ids = [name + " (single hypothesis)" for name in get_id_list()]
                     +[name + " (hypothesis list)" for name in get_id_list()])
def analysis_and_parameters(request):
    analysis, parameters = request.param
    return (analysis, parameters)

@pytest.fixture(scope="module")
def analysis(analysis_and_parameters):
    analy, pars = analysis_and_parameters
    return analy

@pytest.fixture(scope="module")
def parameters(analysis_and_parameters):
    analy, pars = analysis_and_parameters
    return {analy.name: pars}

# Fixture to create tensorflow_probability models from each Analysis/test parameter combination
@pytest.fixture(scope="module")
def model(analysis,parameters):
    return analysis.tensorflow_model(parameters[analysis.name])

# Fixture to create JointDistribution objects from each Analysis/test parameter combination
@pytest.fixture(scope="module")
def joint(analysis,parameters):
    joint = JointDistribution([analysis],parameters)
    return joint

# Fixture to create samples of various shapes
shapes = [1,10,(1,),(10,),(5,3),(5,2,3),(1,5)]
@pytest.fixture(scope="module",params=shapes,ids=["sample_shape={0}".format(s) for s in shapes])
def samples_and_shapes(request,joint):
    sample_shape = request.param
    n = sample_shape
    x = joint.sample(n)
    return n, x 

@pytest.fixture(scope="module")
def samples(samples_and_shapes):
    n,x = samples_and_shapes
    return x

@pytest.fixture(scope="module")
def sample_shape(samples_and_shapes):
    n,x = samples_and_shapes
    return n

@pytest.fixture(scope="module")
def batch_shape(joint):
    # Check for consistent (broadcasted) batch_dims
    shape = joint.bcast_batch_shape_tensor()
    if -1 in shape:
        raise ValueError("-1 detected in batch_shape (shape was {0})".format(shape))
    return shape

@pytest.fixture(scope="module")
def log_prob_shape(sample_shape,batch_shape):
    # Expected shape of log_prob output
    try:
        list_sample_shape = list(sample_shape)
    except TypeError: # single int shapes cause this
        list_sample_shape = [sample_shape]
    p_shape_expected = list_sample_shape + list(batch_shape)
    return p_shape_expected

def test_tensorflow_models(model):
    """tensorflow_model function should return dictionary of tensorflow_probability distribution objects"""
    assert isinstance(model,dict)
    for key,value in model.items():
        assert isinstance(key, str)
        assert isinstance(value, tfd.Distribution)

# Decorator to help run test for all analyses
def do_for_all_analyses(test_func):
    @pytest.mark.parametrize(
        "analysis,single_hypothesis,many_hypotheses",
        zip(get_obj().values(),get_test_hypothesis().values(),get_hypothesis_lists().values()),
        ids = get_id_list()
        )
    def wrapper(analysis,single_hypothesis,many_hypotheses):
        test_func(analysis,single_hypothesis,many_hypotheses)
    return wrapper

def test_JointDistribution_init_basic(analysis):
    """Test that JointDistribution objects can be instantiated from the test analysis objects
       No parmeter version"""
    joint = JointDistribution([analysis])

def test_JointDistribution_init_pars(joint):
    """Test that JointDistribution objects can be instantiated from the test analysis objects
       Fixed parameters version"""
    pass # Just testing that "joint" is created so far.

def test_JointDistribution_output_shapes_single(joint,samples,log_prob_shape):
    """Test that flow of shapes through JointDistribution samples -> pdf is correct"""
    x = samples
    log_prob = joint.log_prob(x)
    # Should obey tensorflow_probability shape rules here.
    # See shapes_readme.md

    # Will only see these print statements when test fails
    print("x:", x)
    print("log_prob:", log_prob)
    print("log_prob_shape:", log_prob_shape)
    assert log_prob.shape == log_prob_shape

def test_sample_shapes(joint,samples):
    """Test that shapes are consistent across simulated, observed, and Asimov data
       for each analysis"""

    print("Simulated samples:", samples)
    print("Observed samples:", joint.Osamples)
    print("Asimov samples:", joint.Asamples)

    for name,eshape in joint.event_shapes().items():
        print("Distribution:", name)
        n = len(eshape)
        print("n:",n)
        print("event shape:", eshape)
        print("Osamples shape:", joint.Osamples[name].shape)
        ashape = joint.Asamples[name].shape
        print("Asamples shape:", ashape)
        print("Asamples inferred event shape:", ashape[-n or len(ashape):])
        sshape = samples[name].shape
        print("Generated samples shape:", sshape)
        print("Generated samples inferred event shape:", sshape[-n or len(sshape):])
        assert joint.Osamples[name].shape == eshape 
        assert ashape[-n or len(ashape):] == eshape
        assert sshape[-n or len(sshape):] == eshape

def test_fit_all(joint,samples,log_prob_shape):
    """Test fitting of all free parameters"""
    log_prob, joint_fitted, par_dict = joint.fit_all(samples)
    print("samples:", samples)
    print("log_prob:", log_prob)
    print("par_dict:", par_dict)
    assert log_prob.shape == log_prob_shape

def test_fit_nuisance(joint,samples,parameters,log_prob_shape):
    """Test fitting of nuisance parameters"""
    log_prob, joint_fitted, par_dict = joint.fit_nuisance(samples,parameters)
    print("samples:", samples)
    print("parameters:", parameters)
    print("log_prob:", log_prob)
    print("par_dict:", par_dict)
    assert log_prob.shape == log_prob_shape

def test_fit_all_numeric(joint,samples):
    """Ensure that exact MLE fits match numerical results"""
    log_prob, joint_fitted, par_dict = joint.fit_all(samples)
    print("samples:", samples)
    print("log_prob:", log_prob)
    print("par_dict:", par_dict)
    log_prob_n, joint_fitted_n, par_dict_n = joint.fit_all(samples,force_numeric=True)
    print("log_prob_n:", log_prob_n)
    print("par_dict_n:", par_dict_n)
    tol = 1e-3
    print("tol = ", tol)
    assert np.all(np.abs(log_prob - log_prob_n)<tol)
    for ka,a in par_dict['all'].items():
        for kp,p in a.items():
            print("Testing par {0} of analysis {1}".format(kp,ka))
            print("p - par_dict_n['all'][ka][kp] :",p - par_dict_n['all'][ka][kp])
            if np.all(p==0): 
                assert np.all((p - par_dict_n['all'][ka][kp]) < tol)
            else: 
                print("np.abs((p - par_dict_n['all'][ka][kp])/p :", np.abs((p - par_dict_n['all'][ka][kp])/p))
                assert np.all(np.abs((p - par_dict_n['all'][ka][kp])/p)<tol) 

def test_fit_nuisance_numeric(joint,samples,parameters):
    """Ensure that exact MLE fits match numerical results"""
    log_prob, joint_fitted, par_dict = joint.fit_nuisance(samples,parameters)
    print("samples:", samples)
    print("parameters:", parameters)
    print("log_prob:", log_prob)
    print("par_dict:", par_dict)
    log_prob_n, joint_fitted_n, par_dict_n = joint.fit_nuisance(samples,parameters,force_numeric=True)
    print("log_prob_n:", log_prob_n)
    print("par_dict_n:", par_dict_n)
    tol = 1e-6
    print("tol = ", tol)
    assert np.all(np.abs(log_prob - log_prob_n)<tol)
    for ka,a in par_dict['all'].items():
        for kp,p_in in a.items():
            p = p_in.numpy()
            p_n = par_dict_n['all'][ka][kp].numpy()
            print("Testing par {0} of analysis {1}".format(kp,ka))
            print("p:", p)
            print("p_n:", p_n)
            print("p - p_n :",p - p_n)
            # Need to deal with cases where p==0 differently due to divide by zero.
            # Kind of hard to test actually since reasonable scaling is hard to detect. TODO: might need to revisit this.
            m0 = p==0
            print("m0:",m0)
            diff = np.abs(p - p_n)
            print("diff:", diff)
            if p.shape==():
                # Scalar cases, cannot use mask
                if m0: 
                    print("np.all(diff < tol):",np.all(diff < tol))
                    assert np.all(diff < tol)
                else:  
                    print("np.all(diff/|p| < tol):",np.all(diff/np.abs(p) < tol))
                    assert np.all(diff/np.abs(p) < tol)
            else:
                # Array cases
                print("np.all(diff[m0] < tol):",np.all(diff[m0] < tol))
                print("np.all(diff[~m0]/|p[~m0]|) < tol):",np.all(diff[~m0]/np.abs(p[~m0]) < tol))
                assert np.all(diff[m0] < tol)
                assert np.all(diff[~m0]/np.abs(p[~m0]) < tol)
 

 
