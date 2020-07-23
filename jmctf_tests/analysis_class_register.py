"""Registration of Analysis classes for automated testing

   TODO: explain how to add new classes here
"""

# From the analysis-specific unit tests we can
# acquire functions to create Analysis objects
# for testing.
from jmctf_tests.unit_tests import test_normalanalysis
from jmctf_tests.unit_tests import test_binnedanalysis

# Add your new analysis unit test module here to include it in all these tests
# It must have 'get_obj' and 'get_model' functions that instantiate a test
# object of your new class, and retrieve the tensorflow_model for your analysis,
# respectively.
analysis_tests = [("NormalAnalysis",test_normalanalysis), 
                  ("BinnedAnalysis",test_binnedanalysis)
                 ]
