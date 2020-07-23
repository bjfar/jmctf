"""Integration tests that apply to all Analysis classes,
   and check that they conform correctly to the
   required interface"""

import pytest

from tensorflow_probability import distributions as tfd
from jmctf import JointDistribution
from jmctf_tests.analysis_class_register import analysis_tests

id_list = [name for name,a in analysis_tests]

def get_all_obj():
    return [a.get_obj() for name,a in analysis_tests]

def get_test_hypothesis():
    return [a.get_single_hypothesis() for name,a in analysis_tests]

def get_hypothesis_lists():
    return [a.get_three_hypotheses() for name,a in analysis_tests]

# Fixture to create just analysis objects
@pytest.fixture(scope="module", 
                params=get_all_obj(),
                ids = id_list)
def analysis(request):
    yield request.param

# Fixture to create analysis objects and provide test parameters to use with them 
@pytest.fixture(scope="module", 
                params=list(zip(get_all_obj(),get_test_hypothesis()))
                      +list(zip(get_all_obj(),get_hypothesis_lists())),
                ids = [name + " (single hypothesis)" for name in id_list]
                     +[name + " (hypothesis list)" for name in id_list])
def analysis_and_parameters(request):
    analysis, parameters = request.param
    yield (analysis, parameters)

# Fixture to create tensorflow_probability models from each Analysis/test parameter combination
@pytest.fixture(scope="module")
def model(analysis_and_parameters):
    analysis, pars = analysis_and_parameters
    yield analysis.tensorflow_model(pars)

# Fixture to create JointDistribution objects from each Analysis/test parameter combination
@pytest.fixture(scope="module")
def joint(analysis_and_parameters):
    analysis, pars = analysis_and_parameters
    joint = JointDistribution([analysis],{analysis.name: pars})
    yield joint

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
        zip(get_all_obj(),get_test_hypothesis(),get_hypothesis_lists()),
        ids = id_list
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

def test_JointDistribution_output_shapes_single(analysis_and_parameters):
    """Test that flow of shapes through JointDistribution samples -> pdf is correct"""
    analysis, pars = analysis_and_parameters
    joint = JointDistribution([analysis],{analysis.name: pars})
    x = joint.sample(10)
    p = joint.log_prob(x)
    print(analysis.name, "pars:", pars)
    print(analysis.name, "x:", x)
    print(analysis.name, "p:", p)
    assert False # Make test fail to see print statements