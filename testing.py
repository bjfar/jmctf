from analysis import collider_analyses_from_long_YAML
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import massminimize as mm

N = int(1e1)

print("Starting...")

# Load analysis metadata from YAML and construct helper ColliderAnalysis objects for it
f = 'old_junk/CBit_analyses.yaml'
stream = open(f, 'r')
analyses_read, SR_map = collider_analyses_from_long_YAML(stream,replace_SR_names=True)
stream.close()

def get_null_model():
    null_dists = []
    for a in analyses_read:
        signal = {sr: 0 for sr in a.SR_names}
        null_dists += a.tensorflow_null_model(signal)
    return tfd.JointDistributionSequential(null_dists)

def get_free_model(pars):
    free_dists = []
    for a in analyses_read:
        signal = {sr: 0 for sr in a.SR_names}
        free_dists += a.tensorflow_free_model(signal,pars[a.name])
    return tfd.JointDistributionSequential(free_dists)

def get_free_parameters():
    pars = {}
    for a in analyses_read:
        pars[a.name] = a.get_tensorflow_variables()
    return pars

joint0 = get_null_model()

# Generate background-only pseudodata to be fitted
samples0 = joint0.sample(N)
 
#print(samples0)

# Define loss function to be minimized
def glob_chi2(pars,data):
    joint_s = get_free_model(pars)
    return -2*joint_s.log_prob(data), None, None

# Get dictionary of tensorflow variables for input into optimizer
pars = get_free_parameters()
#print("Flatten pars list:", list(flatten(pars)))
mm.optimize(pars,mm.tools.func_partial(glob_chi2,data=samples0),step=0.01,tol=0.1,grad_tol=1e-4)

