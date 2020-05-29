"""Simple test example of typical JMCTF pipeline
   Used in the docs
"""

import numpy as np
import tensorflow as tf
from JMCTF import NormalAnalysis, BinnedAnalysis, JointDistribution

# make_norm
norm = NormalAnalysis("Test normal", 5, 2)
# make_binned
# (name, n, b, sigma_b)
bins = [("SR1", 10, 9, 2),
        ("SR2", 50, 55, 4)]
binned = BinnedAnalysis("Test binned", bins)
# make_joint
joint = JointDistribution([norm,binned])
# get_structure
print("Sample structure:",joint.get_sample_structure()) 
# >> {'Test normal': {'x': 1, 'x_theta': 1}, 'Test binned': {'n': 2, 'x': 2}} 
# fit
my_sample = {'Test normal::x': 4.3, 'Test normal::x_theta': 0, 'Test binned::n': [9,53], 'Test binned::x': [0,0]}
# Convert standard numeric types to TensorFlow objects (must be float32)
#my_sample = {k1: {k2: tf.constant(x,dtype="float32") for k2,x in inner.items()} for k1,inner in my_sample.items()} 
q, joint_fitted, all_pars = joint.fit_all(my_sample)
print("q:", q)
print(all_pars)
# The output is not so pretty because the parameters are TensorFlow objects
# We can convert them to numpy for better viewing:
def to_numpy(d):
    out = {}
    for k,v in d.items():
        if isinstance(v, dict): out[k] = to_numpy(v)
        else: out[k] = v.numpy()
    return out
print(to_numpy(all_pars))
# sample
samples = joint_fitted.sample(3)
#print("samples:", samples)
print("samples:",to_numpy(samples))
q_3, joint_fitted_3, all_pars_3 = joint.fit_all(samples)
print(to_numpy(all_pars_3))

# null model
# Learn parameters that we are required to supply
free, fixed, nuis = joint.get_parameter_structure()
print("free:", free)

null = {'Test normal': {'mu': np.array([[0.]],dtype='float32'), 'nuisance': None}, 'Test binned': {'s': np.array([[0., 0.]],dtype='float32'), 'nuisance': None}}
joint_null = joint.fix_parameters(null)
# or alternatively one can supply parameters to the constructor:
# joint_null = JointDistribution([norm,binned],null)
samples = joint_null.sample(3)
q_null, joint_fitted_null, all_pars_null = joint_null.fit_all(samples)
print(to_numpy(all_pars_null))
# Inspect shapes
print({k1: {k2: v2.shape for k2,v2 in v1.items()} for k1,v1 in to_numpy(all_pars_null).items()})
