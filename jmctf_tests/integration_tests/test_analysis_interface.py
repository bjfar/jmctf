"""Integration tests that apply to all Analysis classes,
   and check that they conform correctly to the
   required interface"""

import pytest

from tensorflow_probability import distributions as tfd
from jmctf import JointDistribution
from jmctf_tests.analysis_class_register import get_id_list, get_obj, get_test_hypothesis, get_hypothesis_lists

# Fixture to create just analysis objects
@pytest.fixture(scope="module", 
                params=get_obj().values(),
                ids = get_id_list())
def analysis(request):
    return request.param

# Fixture to create analysis objects and provide test parameters to use with them 
@pytest.fixture(scope="module", 
                params=list(zip(get_obj().values(),get_test_hypothesis().values()))
                      +list(zip(get_obj().values(),get_hypothesis_lists().values())),
                ids = [name + " (single hypothesis)" for name in get_id_list()]
                     +[name + " (hypothesis list)" for name in get_id_list()])
def analysis_and_parameters(request):
    analysis, parameters = request.param
    return (analysis, parameters)

# Fixture to create tensorflow_probability models from each Analysis/test parameter combination
@pytest.fixture(scope="module")
def model(analysis_and_parameters):
    analysis, pars = analysis_and_parameters
    return analysis.tensorflow_model(pars)

# Fixture to create JointDistribution objects from each Analysis/test parameter combination
@pytest.fixture(scope="module")
def joint(analysis_and_parameters):
    analysis, pars = analysis_and_parameters
    joint = JointDistribution([analysis],{analysis.name: pars})
    return joint

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

# Run test for various sample shapes
shapes = [1,10,(1,),(10,),(5,3),(5,2,3),(1,5)]
@pytest.mark.parametrize("sample_shape",shapes,ids=[str(s) for s in shapes])
def test_JointDistribution_output_shapes_single(analysis_and_parameters,sample_shape):
    """Test that flow of shapes through JointDistribution samples -> pdf is correct"""
    analysis, pars = analysis_and_parameters
    joint = JointDistribution([analysis],{analysis.name: pars})
    n = sample_shape
    x = joint.sample(n)
    p = joint.log_prob(x)
    # Should obey tensorflow_probability shape rules here.
    # See shapes_readme.md

    # Check for consistent (broadcasted) batch_dims
    batch_shape = joint.bcast_batch_shape_tensor()

    try:
        list_sample_shape = list(sample_shape)
    except TypeError: # single int shapes cause this
        list_sample_shape = [sample_shape]
    p_shape_expected = list_sample_shape + list(batch_shape)

    # Will only see these print statements when test fails
    print(analysis.name, "pars:", pars)
    print(analysis.name, "x:", x)
    print(analysis.name, "p:", p)
    print(analysis.name, "sample_shape:", sample_shape)
    print(analysis.name, "batch_shape:", batch_shape)
    print(analysis.name, "p_shape_expected:", p_shape_expected)

    assert p.shape == p_shape_expected
