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
joint = JointDistribution([binned])
#joint = JointDistribution([binned])
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

null = {'Test normal': {'mu': [0.], 'nuisance': None}, 'Test binned': {'s': [(0., 0.)], 'nuisance': None}}
joint_null = joint.fix_parameters(null)
# or alternatively one can supply parameters to the constructor:
# joint_null = JointDistribution([norm,binned],null)
samples = joint_null.sample(3)
q_fit, joint_fitted_null, all_pars_null = joint_null.fit_all(samples)
print(to_numpy(all_pars_null))
# Inspect shapes
print({k1: {k2: v2.shape for k2,v2 in v1.items()} for k1,v1 in to_numpy(all_pars_null).items()})

# Ramp it up, plot all samples, plot MLEs.
samples = joint_null.sample(1e6)
shapes = {k: v.shape for k,v in samples.items()}
print(shapes)
q_fit, joint_fitted_null, all_pars_null = joint.fit_all(samples)

#import matplotlib.pyplot as plt
#from JMCTF.plotting import plot_sample_dist
#fig = plot_sample_dist(samples)
##plt.show()
#fig.tight_layout()
#fig.savefig("quickstart_sample_dists.svg")

#from JMCTF.plotting import plot_MLE_dist
#q_null, joint_fitted_null, all_pars_null = joint_null.fit_all(samples)
#print(all_pars_null)
#fig = plot_MLE_dist(all_pars_null)
#fig.tight_layout()
#fig.savefig("quickstart_MLE_dists.svg")

print("Fitting null hypothesis")
q_null, joint_fitted_nuis, pars_nuis = joint.fit_nuisance(null, samples)
LLR = q_null - q_fit
print("LLR:",LLR)

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow_probability import distributions as tfd
fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111)
ax.set_xlabel("LLR")
ax.set(yscale="log")
sns.distplot(LLR, color='b', kde=False, ax=ax, norm_hist=True, label="JMCTF")
q = np.linspace(0, np.max(LLR),1000)
chi2 = tf.math.exp(tfd.Chi2(df=4).log_prob(q)) 
ax.plot(q,chi2,color='b',lw=2,label="chi^2 (DOF=4)")
ax.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
fig.tight_layout()
fig.savefig("quickstart_LLR.svg")
