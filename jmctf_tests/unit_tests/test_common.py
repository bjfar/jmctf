"""Tests for functions in the 'common' library for jmctf"""

import pytest

from tensorflow_probability import distributions as tfd
from jmctf import JointDistribution
import jmctf.common as c
from jmctf_tests.analysis_class_register import get_id_list, get_obj, get_test_hypothesis, get_hypothesis_lists

# Fixture to create analysis objects and provide test parameters to use with them 
@pytest.fixture(scope="module", 
                params=list(zip(get_obj().values(),get_test_hypothesis().values()))
                      +list(zip(get_obj().values(),get_hypothesis_lists().values())),
                ids = [name + " (single hypothesis)" for name in get_id_list()]
                     +[name + " (hypothesis list)" for name in get_id_list()])
def analysis_and_parameters(request):
    analysis, parameters_shapes = request.param
    return (analysis, parameters_shapes)

@pytest.fixture(scope="module")
def analysis(analysis_and_parameters):
    analy, pars = analysis_and_parameters
    return analy

@pytest.fixture(scope="module")
def parameters(analysis_and_parameters):
    analy, (pars, bs, dist_bs) = analysis_and_parameters
    return {analy.name: pars}

@pytest.fixture(scope="module")
def batch_shapes(analysis_and_parameters):
    analy, (pars, batch_shapes, dist_batch_shape) = analysis_and_parameters
    return batch_shapes, dist_batch_shape

# Fixture to create tensorflow_probability models from each Analysis/test parameter combination
@pytest.fixture(scope="module")
def model(analysis,parameters):
    return analysis.tensorflow_model(parameters[analysis.name])

# Fixture to create JointDistribution objects from each Analysis/test parameter combination
@pytest.fixture(scope="module")
def joint(analysis,parameters):
    joint = JointDistribution([analysis],parameters)
    return joint

# Fixture to obtain concatenated parameters for each Analysis/test parameter combination
@pytest.fixture(scope="module")
def par_tensor_and_batch_shape(joint,parameters):
    shapes = joint.parameter_shapes()
    par_tensor, batch_shape, col_list = c.cat_pars_to_tensor(parameters,shapes)
    print("column list:", col_list)
    return par_tensor, batch_shape

def test_cat_pars_to_tensor(par_tensor_and_batch_shape):
    print("par_tensor:", par_tensor_and_batch_shape[0])
    print("batch_shape:", par_tensor_and_batch_shape[1])

def test_decat_tensor_to_pars(par_tensor_and_batch_shape,joint,parameters):
    par_tensor, batch_shape = par_tensor_and_batch_shape 
    shapes = joint.parameter_shapes()
    pars = c.decat_tensor_to_pars(par_tensor,parameters,shapes,batch_shape)
    print("parameters (template)", parameters)
    print("pars:", pars)

def test_all_dist_batch_shape(joint, parameters, batch_shapes):
    exp_batch_shape = batch_shapes[0]
    exp_dist_batch_shape = batch_shapes[1]
    print("joint:", joint)
    shapes = joint.parameter_shapes()     
    print("parameters:", parameters)
    print("parameter shapes:", shapes)
    print("expected distribution batch shape:", exp_dist_batch_shape)
    dist_batch_shape = c.all_dist_batch_shape(parameters, shapes)
    print("inferred distribution batch shape:", dist_batch_shape)
    assert exp_dist_batch_shape == dist_batch_shape

def test_dist_batch_shape(joint, parameters, batch_shapes):
    # Bascially the same as test_all_dist_batch_shape since our
    # tests currently only use one analysis at a time, but we
    # dig out that one analysis manually first.
    exp_batch_shape = batch_shapes[0]
    exp_dist_batch_shape = batch_shapes[1]
    print("joint:", joint)
    shapes = joint.parameter_shapes()     
    print("parameters:", parameters)
    print("parameter shapes:", shapes)
    print("expected distribution batch shape:", exp_dist_batch_shape)
    for analysis_name, par_dict in parameters.items(): 
        analysis_batch_shape = c.dist_batch_shape(par_dict, shapes[analysis_name])
        print("inferred analysis batch shape:", analysis_batch_shape)
        assert exp_dist_batch_shape == analysis_batch_shape # TODO: May need to generalise in future
     
def test_bcast_sample_batch_shape():
    bcast_samples = c.bcast_sample_batch_shape(event_shapes, dist_batch_shape, samples=samples)
 
