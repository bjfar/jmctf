from analysis import collider_analyses_from_long_YAML
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import massminimize as mm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

N = int(1e4)

print("Starting...")

# Load analysis metadata from YAML and construct helper ColliderAnalysis objects for it
f = 'old_junk/CBit_analyses.yaml'
stream = open(f, 'r')
analyses_read, SR_map = collider_analyses_from_long_YAML(stream,replace_SR_names=True)
analyses_read = analyses_read[1:10] # For testing
#analyses_read = [analyses_read[1]] # For testing
#aanalyses_read = [a for a in analyses_read if a.name=="CMS_13TeV_2OSLEP_36invfb"] # For testing
stream.close()

s_in = [1,2]
nosignal = {a.name: {sr: tf.constant([0],dtype=float) for sr in a.SR_names} for a in analyses_read}
signal = {a.name: {sr: tf.constant(s_in,dtype=float) for sr in a.SR_names} for a in analyses_read}

def get_null_model(signal):
    null_dists = []
    layouts = []
    counts = []
    Asamples = []
    for a in analyses_read:
        d, layout, count, asimov = a.tensorflow_null_model(signal[a.name])
        null_dists += d
        layouts += [layout]
        counts += [count]
        Asamples += asimov
    return tfd.JointDistributionSequential(null_dists), layouts, counts, Asamples

def get_free_model(pars,signal):
    free_dists = []
    layouts = []
    counts = []
    for a in analyses_read:
        d, layout, count = a.tensorflow_free_model(signal[a.name],pars[a.name])
        free_dists += d
        layouts += [layout]
        counts += [count]
    return tfd.JointDistributionSequential(free_dists), layouts, counts

def get_Asimov_samples(signal):
    samples = []
    for a in analyses_read:
        samples += s.get_Asimov_samples(signal)
    return samples

def get_free_parameters(samples,layouts,counts,signal):
    """Samples vector and layout provided to compute good starting guesses for parameters"""
    pars = {}
    i = 0
    for a,layout,count in zip(analyses_read,layouts,counts):
        X = samples[i:i+count]
        i += count
        pars[a.name] = a.get_tensorflow_variables(X,layout,signal[a.name])
    return pars

joint0, layouts, counts, samplesAb = get_null_model(nosignal)
joint0s, layouts, counts, samplesAsb = get_null_model(signal)

#print("layouts:", layouts)
#print("counts:", counts)

# Generate background-only pseudodata to be fitted
samples0 = joint0.sample(N)

# Generate signal pseudodata to be fitted
samples0s = joint0s.sample(N)

# Determine asymptotic test statistic distributions from Asimov samples
qsbAsb = -2*(joint0s.log_prob(samplesAsb))
qbAsb  = -2*(joint0.log_prob(samplesAsb)) 
qAsb = qsbAsb - qbAsb

qsbAb = -2*(joint0s.log_prob(samplesAb))
qbAb  = -2*(joint0.log_prob(samplesAb)) 
qAb = qsbAb - qbAb

# Define loss function to be minimized
# Use @tf.function decorator to compile target function into graph mode for faster evaluation
# Takes a while to compile the graph at startup, but execution is quite a bit faster
# Turn off for rapid testing, turn on for production.
#@tf.function
def glob_chi2(pars,data,signal=signal):
    joint_s, layouts, counts = get_free_model(pars,signal)
    q = -2*joint_s.log_prob(data)
    total_loss = tf.math.reduce_sum(q)
    return total_loss, None, q

# Get dictionary of tensorflow variables for input into optimizer
#print("Flatten pars list:", list(flatten(pars)))

opts = {"optimizer": "Adam",
        "step": 0.3,
        "tol": 0.1,
        "grad_tol": 1e-4,
        "max_it": 100,
        "max_same": 5
        }

print("Fitting signal hypothesis")
pars = get_free_parameters(samples0,layouts,counts,signal)
pars, qsb = mm.optimize(pars,mm.tools.func_partial(glob_chi2,data=samples0,signal=signal),**opts)
pars_s = get_free_parameters(samples0s,layouts,counts,signal)
pars_s, qsb_s = mm.optimize(pars_s,mm.tools.func_partial(glob_chi2,data=samples0s,signal=signal),**opts)

print("Fitting background-only hypothesis")
parsb = get_free_parameters(samples0,layouts,counts,nosignal)
parsb, qb = mm.optimize(parsb,mm.tools.func_partial(glob_chi2,data=samples0,signal=nosignal),**opts)
parsb_s = get_free_parameters(samples0s,layouts,counts,nosignal)
parsb_s, qb_s = mm.optimize(parsb_s,mm.tools.func_partial(glob_chi2,data=samples0s,signal=nosignal),**opts)

q = qsb - qb
q_s = qsb_s - qb_s

nplots = len(s_in)
fig = plt.figure(figsize=(12,4*nplots))
for i in range(nplots):
    ax1 = fig.add_subplot(nplots,2,2*i+1)
    ax2 = fig.add_subplot(nplots,2,2*i+2)
    ax2.set(yscale="log")

    qb  = q[:,i].numpy()
    qsb = q_s[:,i].numpy()
    if np.sum(np.isfinite(qb)) < 2:
        print("qb mostly nan!")
    if np.sum(np.isfinite(qsb)) < 2:
        print("qsb mostly nan!")
    qb = qb[np.isfinite(qb)]
    qsb = qsb[np.isfinite(qsb)]

    sns.distplot(qb , bins=50, color='b',kde=False, ax=ax1, norm_hist=True, label="s={0}".format(s_in[i]))
    sns.distplot(qsb, bins=50, color='r', kde=False, ax=ax1, norm_hist=True, label="s={0}".format(s_in[i]))

    # Compute and plot asymptotic distributions!
    var_mu_sb = 1/tf.abs(qAsb[i]) 
    var_mu_b  = 1/tf.abs(qAb[i]) 

    Eq_sb = -1 / var_mu_sb
    Eq_b  = 1 / var_mu_b

    Vq_sb = 4 / var_mu_sb
    Vq_b  = 4 / var_mu_b

    qsbx = np.linspace(np.min(qsb),np.max(qsb),1000)
    qsby = tf.math.exp(tfd.Normal(loc=Eq_sb, scale=tf.sqrt(Vq_sb)).log_prob(qsbx)) 
    sns.lineplot(qsbx,qsby,color='r',ax=ax1)
    qbx = np.linspace(np.min(qb),np.max(qb),1000)
    qby = tf.math.exp(tfd.Normal(loc=Eq_b, scale=tf.sqrt(Vq_b)).log_prob(qbx)) 
    sns.lineplot(qbx,qby,color='b',ax=ax1)

    sns.distplot(qb, color='b', kde=False, ax=ax2, norm_hist=True, label="s={0}".format(s_in[i]))
    sns.distplot(qsb, color='r', kde=False, ax=ax2, norm_hist=True, label="s={0}".format(s_in[i]))
    sns.lineplot(qbx, qby,color='b',ax=ax2)
    sns.lineplot(qsbx,qsby,color='r',ax=ax2)

    ax1.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
    ax2.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)

plt.tight_layout()
fig.savefig("qsb_dists.png")


