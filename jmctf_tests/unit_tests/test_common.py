"""Tests for functions in the 'common' library for jmctf"""

import pytest

from tensorflow_probability import distributions as tfd
from jmctf import JointDistribution
import jmctf.common as c
from jmctf_tests.analysis_class_register import get_id_list, get_obj, get_test_hypothesis, get_hypothesis_lists
from jmctf_tests.common_fixtures import analysis, pars, pars0, joint0, samples

# Fixture to create analysis objects and provide test parameters to use with them 
@pytest.fixture(scope="module", 
                params=list(zip(get_obj().values(),get_test_hypothesis().values()))
                      +list(zip(get_obj().values(),get_hypothesis_lists().values())),
                ids = [name + " (single hypothesis)" for name in get_id_list()]
                     +[name + " (hypothesis list)" for name in get_id_list()])
def analysis_and_parameters(request):
    analysis, parameters_shapes = request.param
    return (analysis, parameters_shapes)

# Fixture required by fixtures from common_fixtures.py
@pytest.fixture(scope="module")
def analysis_params(analysis_and_parameters):
    analy, (pars, bs, dist_bs) = analysis_and_parameters
    return analy, pars

@pytest.fixture(scope="module")
def batch_shapes(analysis_and_parameters):
    analy, (pars, batch_shapes, dist_batch_shape) = analysis_and_parameters
    return batch_shapes, dist_batch_shape

# Number of samples to draw during tests
N = 1000
@pytest.fixture(scope="module")
def Nsamples():
    return N

# Fixture to create tensorflow_probability models from each Analysis/test parameter combination
@pytest.fixture(scope="module")
def model(analysis, pars):
    return analysis.tensorflow_model(pars[analysis.name])

# Fixture to create JointDistribution objects from each Analysis/test parameter combination
@pytest.fixture(scope="module")
def joint(analysis, pars):
    joint = JointDistribution([analysis], pars)
    return joint

# Fixture to obtain concatenated parameters for each Analysis/test parameter combination
@pytest.fixture(scope="module")
def par_tensor_and_batch_shape(joint, pars):
    shapes = joint.parameter_shapes()
    par_tensor, batch_shape, col_list = c.cat_pars_to_tensor(pars,shapes)
    print("column list:", col_list)
    return par_tensor, batch_shape

def test_cat_pars_to_tensor(par_tensor_and_batch_shape):
    print("par_tensor:", par_tensor_and_batch_shape[0])
    print("batch_shape:", par_tensor_and_batch_shape[1])

def test_decat_tensor_to_pars(par_tensor_and_batch_shape, joint, pars):
    par_tensor, batch_shape = par_tensor_and_batch_shape 
    shapes = joint.parameter_shapes()
    out_pars = c.decat_tensor_to_pars(par_tensor, pars, shapes, batch_shape)
    print("parameters (template)", pars)
    print("pars:", out_pars)

def test_all_dist_batch_shape(joint, pars, batch_shapes):
    exp_batch_shape = batch_shapes[0]
    exp_dist_batch_shape = batch_shapes[1]
    print("joint:", joint)
    shapes = joint.parameter_shapes()     
    print("parameters:", pars)
    print("parameter shapes:", shapes)
    print("expected distribution batch shape:", exp_dist_batch_shape)
    dist_batch_shape = c.all_dist_batch_shape(pars, shapes)
    print("inferred distribution batch shape:", dist_batch_shape)
    assert exp_dist_batch_shape == dist_batch_shape

def test_dist_batch_shape(joint, pars, batch_shapes):
    # Basically the same as test_all_dist_batch_shape since our
    # tests currently only use one analysis at a time, but we
    # dig out that one analysis manually first.
    exp_batch_shape = batch_shapes[0]
    exp_dist_batch_shape = batch_shapes[1]
    print("joint:", joint)
    shapes = joint.parameter_shapes()     
    print("parameters:", pars)
    print("parameter shapes:", shapes)
    print("expected distribution batch shape:", exp_dist_batch_shape)
    for analysis_name, par_dict in pars.items(): 
        analysis_batch_shape = c.dist_batch_shape(par_dict, shapes[analysis_name])
        print("inferred analysis batch shape:", analysis_batch_shape)
        assert exp_dist_batch_shape == analysis_batch_shape # TODO: May need to generalise in future
    
def test_make_samples_orthogonal(joint, pars, samples):
    event_shapes = joint.event_shapes() 
    par_shapes = joint.parameter_shapes()
    dist_batch_shape = c.all_dist_batch_shape(pars, par_shapes)
    #print("  parameters:", pars)
    print("  samples:", samples)
    print("  event_shapes:", event_shapes)
    print("  par_shapes:", par_shapes)    
    ortho_samples = c.make_samples_orthogonal(event_shapes, dist_batch_shape, samples)
    print("  orthogonal samples:", ortho_samples)
    # TODO need some test data/calculation to compare with

def test_bcast_sample_batch_shape(joint, pars, samples):
    """For this test we will treat the samples as completely orthogonal to
       the parameters, that is, we suppose we want (say) a log_probability
       for every combination of samples and parameters.
       Other broadcasting is possible, but then shapes need to match up
       in specific ways, so we need tests of specific scenarios rather
       than the generic test done here. See shapes_readme.md.
    """
    
    event_shapes = joint.event_shapes() 
    par_shapes = joint.parameter_shapes()
    dist_batch_shape = c.all_dist_batch_shape(pars, par_shapes)
    print("  parameters:", pars)
    print("  samples:", samples)
    print("  event_shapes:", event_shapes)
    print("  par_shapes:", par_shapes)
    print("  dist_batch_shape:", dist_batch_shape)        
    bcast_samples = c.bcast_sample_batch_shape(event_shapes, dist_batch_shape, samples=samples)
    print("  bcast samples:", bcast_samples)

