"""Unit tests for NormalAnalysis class"""

from JMCTF.normal_analysis import NormalAnalysis

name = "test_name"
x_obs = 5
sigma = 1

def get_obj():
    return NormalAnalysis(name,x_obs,sigma)

def test_NormalAnalysis_init():
    obj = get_obj() 
    assert obj.name == name
    assert obj.x_obs == x_obs
    assert obj.sigma == sigma

def test_NormalAnalysis_tensorflow_model():
    obj = get_obj()
    pars = {'mu': 1.5, 
            'theta': 0.5,
            'sigma_t': 1.}
    model = obj.tensorflow_model(pars)
    assert "{0}::x".format(name) in model.keys()
    assert "{0}::x_theta".format(name) in model.keys()


