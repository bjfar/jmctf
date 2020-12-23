"""Unit tests for BinnedAnalysis class

   This version sets up an analysis with just one bin,
   for the simplest test scenario possible.
"""

import numpy as np
import tensorflow as tf
import jmctf.common as c
from jmctf.binned_analysis import BinnedAnalysis

name = "test_binned_single"
#bins = [("SR1", 100, 90, 20)] # name, obs, b, sigma_b
bins = [("SR1", 10, 200, 5)]

# Functions required for compatibility with generic tests
# -------------------------------------------------------
def get_obj():
    return BinnedAnalysis(name,bins)

def get_single_hypothesis():
    pars = {'s': tf.constant((0,),dtype=c.TFdtype), 
            'theta': tf.constant((0,),dtype=c.TFdtype)
            }
    return pars

def get_three_hypotheses():
    pars = {'s': tf.constant([(0,),(1,),(2,)],dtype=c.TFdtype), 
            'theta': tf.constant((0,),dtype=c.TFdtype)
            }
    return pars

def get_hypothesis_curves():
    """Curves of hypotheses that should almost always encompass the best-fit point
       with sufficient density to plot reasonably smooth log-likelihood curves.
       Returns a dict, in case multiple curves in different directions needed.
       (keys used to name output tests/plots)"""
    sr1 = [(x,) for x in np.linspace(0,20,10)]
    pars = {'s': tf.constant(sr1,dtype=c.TFdtype),
            'theta': tf.constant((0,),dtype=c.TFdtype)}
    return {("s",0): pars} 

# -------------------------------------------------------

def get_model():
    obj = get_obj()
    pars = get_single_hypothesis()
    model = obj.tensorflow_model(pars)
    return model

def test_BinnedAnalysis_init():
    obj = get_obj() 
    for i,sr in enumerate(bins):
        assert obj.SR_names[i] == sr[0]
        assert obj.SR_n[i]     == sr[1]
        assert obj.SR_b[i]     == sr[2]
        assert obj.SR_b_sys[i] == sr[3]

def test_BinnedAnalysis_tensorflow_model():
    model = get_model()
    assert "n" in model.keys()
    assert "x" in model.keys() # Uncorrelated case
