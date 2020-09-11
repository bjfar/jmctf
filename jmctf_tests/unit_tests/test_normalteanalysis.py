"""Unit tests for NormalTEAnalysis class"""

import numpy as np
import tensorflow as tf
import jmctf.common as c
from jmctf.normalte_analysis import NormalTEAnalysis

name = "test_normalte"
x_obs = 5
sigma = 1

# Functions required for compatibility with generic tests
# -------------------------------------------------------
def get_obj():
    return NormalTEAnalysis(name,x_obs,sigma)

def get_single_hypothesis():
    pars = {'mu': tf.constant(1.5,dtype=c.TFdtype),
            'theta': tf.constant(0.5,dtype=c.TFdtype),
            'sigma_t': tf.constant(1.,dtype=c.TFdtype)}
    return pars

def get_three_hypotheses():
    pars = {'mu': tf.constant([0.,1.,2.],dtype=c.TFdtype),
            'theta': tf.constant([0.,0.,0.],dtype=c.TFdtype),
            'sigma_t': tf.constant([1e-10,0.5,1.],dtype=c.TFdtype)} # Currently 0 doesn't work due to divide by zero
    return pars

def get_hypothesis_curves():
    """Curves of hypotheses that should almost always encompass the best-fit point
       with sufficient density to plot reasonably smooth log-likelihood curves.
       Returns a dict, in case multiple curves in different directions needed.
       (keys used to name output tests/plots)"""
    pars = {'mu': tf.constant(np.linspace(0,10,20),dtype=c.TFdtype),
            'theta': tf.constant(0.,dtype=c.TFdtype),
            'sigma_t': tf.constant(1.,dtype=c.TFdtype)} # Currently 0 doesn't work due to divide by zero
    return {"mu": pars}

# -------------------------------------------------------

def get_model():
    obj = get_obj()
    pars = get_single_hypothesis()
    model = obj.tensorflow_model(pars)
    return model

def test_NormalTEAnalysis_init():
    obj = get_obj() 
    assert obj.name == name
    assert obj.x_obs == x_obs
    assert obj.sigma == sigma

def test_NormalTEAnalysis_tensorflow_model():
    model = get_model()
    assert "x" in model.keys()
    assert "x_theta" in model.keys()


