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

def get_all_models():
    return [a.get_model() for name,a in analysis_tests]

def get_test_hypothesis():
    return [a.get_single_hypothesis() for name,a in analysis_tests]

def get_hypothesis_lists():
    return [a.get_three_hypotheses() for name,a in analysis_tests]

def test_tensorflow_models():
    """tensorflow_model function should return dictionary of tensorflow_probability distribution objects"""
    for m in get_all_models():
        assert isinstance(m,dict)
        for key,value in m.items():
            assert isinstance(key, str)
            assert isinstance(value, tfd.Distribution)

# Decorator to help run test for all analyses
def do_for_all(test_func):
    @pytest.mark.parametrize(
        "analysis,single_hypothesis,many_hypotheses",
        zip(get_all_obj(),get_test_hypothesis(),get_hypothesis_lists()),
        ids = id_list
        )
    def wrapper(analysis,single_hypothesis,many_hypotheses):
        test_func(analysis,single_hypothesis,many_hypotheses)
    return wrapper

@do_for_all
def test_JointDistribution_init_basic(analysis,single_hypothesis,many_hypotheses):
    """Test that JointDistribution objects can be instantiated from the test analysis objects
       No parmeter version"""
    joint = JointDistribution([analysis])

@do_for_all
def test_JointDistribution_init_single(analysis,single_hypothesis,many_hypotheses):
    """Test that JointDistribution objects can be instantiated from the test analysis objects
       Single (set of) parameters version"""
    joint = JointDistribution([analysis],{analysis.name: single_hypothesis})

@do_for_all
def test_JointDistribution_init_many(analysis,single_hypothesis,many_hypotheses):
    """Test that JointDistribution objects can be instantiated from the test analysis objects
       Many (sets of) parameters version"""
    joint = JointDistribution([analysis],{analysis.name: many_hypotheses})

#def test_model_output_shapes():
#    """Test that flow of shapes through model samples -> pdf is correct"""
#    for m,hlist in zip(get_all_models(),get_hypothesis_lists()):
        
        


