"""Simple test example of typical JMCTF pipeline
   Used in the docs
"""

import numpy as np
import tensorflow as tf
from JMCTF import NormalAnalysis, BinnedAnalysis, JointDistribution

verb = True

# make_norm
sigma = 2.
norm = NormalAnalysis("Test normal", 5, sigma)
# make_binned
# (name, n, b, sigma_b)
bins = [("SR1", 10, 9, 2),
        ("SR2", 50, 55, 4)]
binned = BinnedAnalysis("Test binned", bins)
# make_joint
joint = JointDistribution([norm,binned])
#joint = JointDistribution([binned])
#joint = JointDistribution([norm])
#DOF = 1
DOF = 3

sig_t = 1.

# get_structure
print("Sample structure:",joint.get_sample_structure()) 
# >> {'Test normal': {'x': 1, 'x_theta': 1}, 'Test binned': {'n': 2, 'x': 2}} 
# fit
my_sample = {'Test normal::x': 4.3, 'Test normal::x_theta': 0, 'Test binned::n': [9,53], 'Test binned::x': [0,0]}
# Convert standard numeric types to TensorFlow objects (must be float32)
#my_sample = {k1: {k2: tf.constant(x,dtype="float32") for k2,x in inner.items()} for k1,inner in my_sample.items()}
fixed_pars={"Test normal": {"sigma_t": sig_t}} # Extra fixed parameter for NormalDist analysis
print("First 'fit_all'")
q, joint_fitted, all_pars, fitted_pars, const_pars = joint.fit_all(my_sample,fixed_pars)
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
samples = joint_fitted.sample(10)
#print("samples:", samples)
print("samples:",to_numpy(samples))

q_3, joint_fitted_3, all_pars_3, fitted_pars_3, const_pars_3 = joint.fit_all(samples,fixed_pars)
print(to_numpy(all_pars_3))

# null model
# Learn parameters that we are required to supply
free, fixed, nuis = joint.get_parameter_structure()
print("free:", free)
print("fixed:", fixed)
print("nuis", nuis)

null = {'Test normal': {'mu': [0.], 'sigma_t': [sig_t], 'test_wrong': [5.]}, 'Test binned': {'s': [(0., 0.)]}}

joint_null = joint.fix_parameters(null)
# or alternatively one can supply parameters to the constructor:
# joint_null = JointDistribution([norm,binned],null)
samples = joint_null.sample(3)
print("***********************")
print("FITTING ALL PARS")
print("***********************")
q_fit, joint_fitted_null, all_pars_null, fitted_pars_null, const_pars_null = joint.fit_all(samples,fixed_pars)
print("samples:", to_numpy(samples))
print("all fitted parameters:", all_pars_null)
print("q_fit:", q_fit)
print("q_recalc:", -2*joint_fitted_null.log_prob(samples))
print("logL parts:", joint_fitted_null.log_prob_parts(samples))

print("parameters:", joint_fitted_null.parameters)
for name,d in joint_fitted_null.parameters['model'].items():
    print("{0} parameters: {1}".format(name,d.parameters))
print("trainable_variables:", joint_fitted_null.trainable_variables)

check=False
if check:
    print("Checking MLEs and q calculations:")
    for i in range(3):
        x = samples["Test normal::x"][i]
        x_theta = samples["Test normal::x_theta"][i]
        mu_MLE = all_pars_null["Test normal"]["mu"][i]
        theta_MLE = all_pars_null["Test normal"]["theta"][i]
        print("x = {0}, x_theta = {1}, x - x_theta = {2}, mu_MLE = {3}, theta_MLE = {4}".format(x,x_theta,x-x_theta,mu_MLE,theta_MLE))
    
        chi2_x = (mu_MLE + theta_MLE - x)**2 / sigma**2
        chi2_xt = (theta_MLE - x_theta)**2 / sig_t**2
        print("chi2_x={0}, chi2_xt={1}, chi2={2}".format(chi2_x,chi2_xt,chi2_x+chi2_xt))
        const_x = np.log(2*np.pi) + 2*np.log(sigma)
        const_xt = np.log(2*np.pi) + 2*np.log(sig_t)
        print("q_x={0}, q_xt={1}, q={2}".format(chi2_x+const_x,chi2_xt+const_xt,chi2_x+chi2_xt+const_x+const_xt))
        print("logl_x={0}, logl_xt={1}, logl={2}".format(-0.5*(chi2_x+const_x),-0.5*(chi2_xt+const_xt),-0.5*(chi2_x+chi2_xt+const_x+const_xt)))

print("***********************")
print("FITTING NUISANCE PARS")
print("***********************")
q_null, joint_fitted_nuis, all_pars_nuis, fitted_pars_nuis, const_pars_nuis = joint.fit_nuisance(samples, null)
print("all_pars_null (3):", to_numpy(all_pars_null))
print("all_pars_nuis (3)    :", to_numpy(all_pars_nuis))

LLR = q_null - q_fit
print("q_fit:", q_fit)
print("q_null:", q_null)
print("LLR:",LLR)

print("============================")

# Inspect shapes
print({k1: {k2: v2.shape for k2,v2 in v1.items()} for k1,v1 in to_numpy(all_pars_null).items()})

# Ramp it up, plot all samples, plot MLEs.
samples = joint_null.sample(1e6)
shapes = {k: v.shape for k,v in samples.items()}
print(shapes)
q_fit, joint_fitted_null, all_pars_null, fitted_pars_null, const_pars_null = joint.fit_all(samples,fixed_pars,verbose=verb)

print("Fitting null hypothesis")
q_null, joint_fitted_nuis, all_pars_nuis, fitted_pars_nuis, const_pars_nuis = joint.fit_nuisance(samples, null, verbose=verb)
LLR = q_null - q_fit
print("q_fit:", q_fit)
print("q_null:", q_null)
print("LLR:",LLR)

print("all_pars_null (1e6):", to_numpy(all_pars_null))
print("all_pars_nuis (1e6)    :", to_numpy(all_pars_nuis))

import matplotlib.pyplot as plt
from JMCTF.plotting import plot_sample_dist, plot_MLE_dist
fig, ax_dict = plot_sample_dist(samples)
#plt.show()
fig.tight_layout()
fig.savefig("quickstart_sample_dists.svg")

fig, ax_dict = plot_MLE_dist(fitted_pars_null)
plot_MLE_dist(fitted_pars_nuis,ax_dict) # Overlay nuis MLE dists onto full MLE dists
fig.tight_layout()
fig.savefig("quickstart_MLE_dists.svg")

#from JMCTF.plotting import plot_MLE_dist
#q_null, joint_fitted_null, all_pars_null = joint_null.fit_all(samples)
#print(all_pars_null)
#fig = plot_MLE_dist(all_pars_null)
#fig.tight_layout()
#fig.savefig("quickstart_MLE_dists.svg")

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow_probability import distributions as tfd
fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111)
ax.set_xlabel("LLR")
ax.set(yscale="log")
sns.distplot(LLR, color='b', kde=False, ax=ax, norm_hist=True, label="JMCTF")
q = np.linspace(0, np.max(LLR),1000)
chi2 = tf.math.exp(tfd.Chi2(df=DOF).log_prob(q)) 
ax.plot(q,chi2,color='b',lw=2,label="chi^2 (DOF={0})".format(DOF))
ax.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
fig.tight_layout()
fig.savefig("quickstart_LLR.svg")
