"""Simple test example of typical JMCTF pipeline
   Used in the docs
"""

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
print(joint.get_sample_structure()) 
# >> {'Test normal': {'x': 1, 'x_theta': 1}, 'Test binned': {'n': 2, 'x': 2}} 
# fit
my_sample = {'Test normal': {'x': 4.3, 'x_theta': 0}, 'Test binned': {'n': [9,53], 'x': [0,0]}}
# Convert standard numeric types to TensorFlow objects (must be float32)
my_sample = {k1: {k2: tf.constant(x,dtype="float32") for k2,x in inner.items()} for k1,inner in my_sample.items()} 
joint.fit_all(my_sample)


