"""Unit tests for NormalAnalysis class"""

import tensorflow as tf
import jmctf.common as c
from jmctf.normal_analysis import NormalAnalysis

name = "test_normal"
x_obs = 5
sigma = 1

def get_obj():
    return NormalAnalysis(name,x_obs,sigma)

def get_model():
    obj = get_obj()
    pars = {'mu': tf.constant(1.5,dtype=c.TFdtype) ,
            'theta': tf.constant(0.5,dtype=c.TFdtype),
            'sigma_t': tf.constant(1.,dtype=c.TFdtype)}
    model = obj.tensorflow_model(pars)
    return model

def test_NormalAnalysis_init():
    obj = get_obj() 
    assert obj.name == name
    assert obj.x_obs == x_obs
    assert obj.sigma == sigma

def test_NormalAnalysis_tensorflow_model():
    model = get_model()
    assert "x" in model.keys()
    assert "x_theta" in model.keys()
