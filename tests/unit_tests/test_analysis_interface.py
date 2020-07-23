"""Unit tests that apply to all Analysis classes,
   and check that they conform correctly to the
   required interface"""

from tensorflow_probability import distributions as tfd

from jmctf.normal_analysis import NormalAnalysis
from jmctf.binned_analysis import BinnedAnalysis

# From the analysis-specific unit tests we can
# acquire functions to create Analysis objects
# for testing.
import test_normalanalysis
import test_binnedanalysis

# Add your new analysis unit test module here to include it in all these tests
# It must have 'get_obj' and 'get_model' functions that instantiate a test
# object of your new class, and retrieve the tensorflow_model for your analysis,
# respectively.
analysis_tests = [test_normalanalysis, test_binnedanalysis]

def get_all_obj():
    return [a.get_obj() for a in analysis_tests]

def get_all_models():
    return [a.get_model() for a in analysis_tests]

def test_tensorflow_models():
    """tensorflow_model function should return dictionary of tensorflow_probability distribution objects"""
    for m in get_all_models():
        assert isinstance(m,dict)
        for key,value in m.items():
            assert isinstance(key, str)
            assert isinstance(value, tfd.Distribution)
