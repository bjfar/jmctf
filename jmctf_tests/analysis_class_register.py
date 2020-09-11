"""Registration of Analysis classes for automated testing

   TODO: explain how to add new classes here
"""

# From the analysis-specific unit tests we can
# acquire functions to create Analysis objects
# for testing.
from jmctf_tests.unit_tests import test_normalanalysis, test_normalteanalysis, test_binnedanalysis, test_binnedanalysis_cov

# Add your new analysis unit test module here to include it in all these tests
# It must have 'get_obj' and 'get_model' functions that instantiate a test
# object of your new class, and retrieve the tensorflow_model for your analysis,
# respectively.
analysis_tests = {"NormalAnalysis": test_normalanalysis, 
                  "NormalTEAnalysis": test_normalteanalysis,
                  "BinnedAnalysis": test_binnedanalysis,
                  "BinnedAnalysis_cov": test_binnedanalysis_cov
                 }

# Useful generic objects for parameterising tests
# Decide to use functions rather than globals to prevent any tests from messing them up

def get_id_list():
    return [name for name in analysis_tests.keys()]

def get_obj(name=None):
    if name is None:
        out = {name: a.get_obj() for name,a in analysis_tests.items()}
    else:
        out = analysis_tests[name].get_obj()
    return out

def get_test_hypothesis(name=None):
    if name is None:
        out = {name: a.get_single_hypothesis() for name,a in analysis_tests.items()}
    else:
        out = analysis_tests[name].get_single_hypothesis()
    return out

def get_hypothesis_lists(name=None):
    if name is None:
        out = {name: a.get_three_hypotheses() for name,a in analysis_tests.items()}
    else:
        out = analysis_tests[name].get_three_hypotheses()
    return out

def get_hypothesis_curves(name=None):
    if name is None:
        out = {name: a.get_hypothesis_curves() for name,a in analysis_tests.items()}
    else:
        out = analysis_tests[name].get_hypothesis_curves()
    return out
