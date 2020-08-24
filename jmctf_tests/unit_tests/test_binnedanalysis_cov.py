"""Unit tests for BinnedAnalysis class
   This version uses a test case with a covariance matrix"""

import tensorflow as tf
import jmctf.common as c
from jmctf.binned_analysis import BinnedAnalysis
from jmctf_tests.unit_tests.test_binnedanalysis import *


name = "test_binned"
bins = [("SR1", 10, 9, 2),
        ("SR2", 50, 55, 4)]
cov = [[2**2,0],
       [0,4**2]] # Uncorrelated, for simplicity
cov_order = "use SR order"

# Should only have to re-write the "get_obj" and "get_model" functions, I think.

# Functions required for compatibility with generic tests
# -------------------------------------------------------
def get_obj():
    return BinnedAnalysis(name,bins,cov,cov_order)
# -------------------------------------------------------

def get_model():
    obj = get_obj()
    pars = get_single_hypothesis()
    model = obj.tensorflow_model(pars)
    return model

def test_BinnedAnalysis_cov_init():
    obj = get_obj() 
    for i,sr in enumerate(bins):
        assert obj.SR_names[i] == sr[0]
        assert obj.SR_n[i]     == sr[1]
        assert obj.SR_b[i]     == sr[2]
        assert obj.SR_b_sys[i] == sr[3]

def test_BinnedAnalysis_cov_tensorflow_model():
    model = get_model()
    assert "n" in model.keys()
    assert "x_cov" in model.keys() # Correlated case

# Make sure that shape output for correlated case matches uncorrelated case
def test_BinnedAnalysis_cov_shape_compatibility():
    pass


