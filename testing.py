from analysis import collider_analyses_from_long_YAML, JMCJoint
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import massminimize as mm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

N = int(5e4)

print("Starting...")

# Load analysis metadata from YAML and construct helper ColliderAnalysis objects for it
f = 'old_junk/CBit_analyses.yaml'
stream = open(f, 'r')
analyses_read, SR_map, iSR_map = collider_analyses_from_long_YAML(stream,replace_SR_names=True)
#print(analyses_read[16].name, SR_map["SR220"]); quit()
#analyses = analyses_read[10:] # For testing
#analyses = [analyses_read[16]] # For testing
#analyses = analyses_read[2:4] # For testing
#analyses = [a for a in analyses_read if a.name=="CMS_13TeV_2OSLEP_36invfb"] # For testing
#analyses = [a for a in analyses_read if a.name=="CMS_13TeV_MultiLEP_2SSLep_36invfb"]
#analyses = [a for a in analyses_read if a.name=="ATLAS_13TeV_3b_discoverySR_24invfb"]
#analyses = [a for a in analyses_read if a.name=="TEST"]
#analyses = [a for a in analyses_read if a.name=="ATLAS_13TeV_RJ3L_3Lep_36invfb"]
#analyses = [a for a in analyses_read if a.name=="CMS_13TeV_2LEPsoft_36invfb"]
analyses = analyses_read
stream.close()

s_in = [.5,1.]
nosignal = {a.name: {'s': tf.constant([[0. for sr in a.SR_names]],dtype=float)} for a in analyses}
signal = {a.name: {'s': tf.constant([[s for sr in a.SR_names] for s in s_in], dtype=float)} for a in analyses}
nullnuis = {a.name: {'nuisance': None} for a in analyses} # Use to automatically set nuisance parameters to zero for sample generation

# # ATLAS_13TeV_RJ3L_3Lep_36invfb
# # best fit signal from MSSMEW analysis, for testing
srs = ["3LHIGH__i0", "3LINT__i1", "3LLOW__i2", "3LCOMP__i3"]
s = [0.2313686860130337, 0.5370300693697021, 3.6212383333783076, 3.268878683119926]
#signal = {"ATLAS_13TeV_RJ3L_3Lep_36invfb": {'s': tf.constant([[si] for si in s],dtype=float)}} # order needs to match a.SR_names!
# n = [2, 1, 20, 12] 
# obs_data = {"ATLAS_13TeV_RJ3L_3Lep_36invfb::{0}::n".format(iSR_map[sr]): tf.constant([ni],dtype=float) for ni,sr in zip(n,srs)}
# obs_data_x = {"ATLAS_13TeV_RJ3L_3Lep_36invfb::{0}::x".format(iSR_map[sr]): tf.constant([0],dtype=float) for sr in srs}
# obs_data.update(obs_data_x)
# 
# print(nosignal)
# print(nullnuis)

def deep_merge(a, b):
    """
    From https://stackoverflow.com/a/56177639/1447953
    Merge two values, with `b` taking precedence over `a`.

    Semantics:
    - If either `a` or `b` is not a dictionary, `a` will be returned only if
      `b` is `None`. Otherwise `b` will be returned.
    - If both values are dictionaries, they are merged as follows:
        * Each key that is found only in `a` or only in `b` will be included in
          the output collection with its value intact.
        * For any key in common between `a` and `b`, the corresponding values
          will be merged with the same semantics.
    """
    if not isinstance(a, dict) or not isinstance(b, dict):
        return a if b is None else b
    else:
        # If we're here, both a and b must be dictionaries or subtypes thereof.

        # Compute set of all keys in both dictionaries.
        keys = set(a.keys()) | set(b.keys())

        # Build output dictionary, merging recursively values with common keys,
        # where `None` is used to mean the absence of a value.
        return {
            key: deep_merge(a.get(key), b.get(key))
            for key in keys
        }

def get_nuis_parameters(analyses,signal,samples):
    """Samples vector and signal provided to compute good starting guesses for parameters"""
    pars = {}
    for a in analyses:
        pars[a.name] = a.get_nuisance_tensorflow_variables(samples,signal[a.name])
    return pars

# Create joint distributions
joint0  = JMCJoint(analyses,deep_merge(nosignal,nullnuis))
joint0s = JMCJoint(analyses,deep_merge(signal,nullnuis))

# Get Asimov samples for nuisance MLEs with fixed signal hypotheses
samplesAb = joint0.Asamples
samplesAsb = joint0s.Asamples 
 
# Get observed data
obs_data = joint0.Osamples

# Define loss function to be minimized
# Use @tf.function decorator to compile target function into graph mode for faster evaluation
# Takes a while to compile the graph at startup, but execution is quite a bit faster
# Turn off for rapid testing, turn on for production.

def glob_chi2(pars,data,signal):
    #print("pars:",pars)
    #print("signal:",signal)
    joint_s = JMCJoint(analyses,deep_merge(pars,signal))
    #print("str(d):", str(joint_s))
    #print("data:",data)
    q = -2*joint_s.log_prob(data)
    #print("q:",q)
    total_loss = tf.math.reduce_sum(q)
    return total_loss, None, q

# Get dictionary of tensorflow variables for input into optimizer
#print("Flatten pars list:", list(flatten(pars)))

opts = {"optimizer": "Adam",
        "step": 0.3,
        "tol": 0.01,
        "grad_tol": 1e-4,
        "max_it": 100,
        "max_same": 5
        }

#samples0 = joint0.sample(N)
#samples0s = joint0s.sample(N)
#print("samples0:", samples0)
#print("samples0s:", samples0s)
#print("samplesAsb:",samplesAsb)
#quit()

# Evaluate distributions for Asimov datasets, in case where we
# know that the MLEs for those samples are the true parameters
qsbAsb = -2*(joint0s.log_prob(samplesAsb))
qbAb  = -2*(joint0.log_prob(samplesAb))
#print("qsbAsb:", qsbAsb)
#print("qbAb:", qbAb)

# Fit distributions for Asimov datasets for the other half of each
# likelihood ratio
print("Fitting w.r.t Asimov samples")
#pars_nosigAb = get_nuis_parameters(analyses,nosignal,samplesAb) # For testing can check these recover the correct parameters and q match the above
#pars_sigAsb  = get_nuis_parameters(analyses,signal,samplesAsb)
#none, qbAb    = mm.optimize(pars_nosigAb, mm.tools.func_partial(glob_chi2,data=samplesAb,signal=nosignal),**opts)
#none, qsbAsb  = mm.optimize(pars_sigAsb,  mm.tools.func_partial(glob_chi2,data=samplesAsb,signal=signal),**opts)
#print("qsbAsb:", qsbAsb)
#print("qbAb:", qbAb)
#print("pars_nosigAb:", pars_nosigAsb)
#print("pars_sigAb:", pars_sigAsb)
pars_nosigAsb = get_nuis_parameters(analyses,nosignal,samplesAsb)
pars_sigAb    = get_nuis_parameters(analyses,signal,samplesAb)
none, qbAsb = mm.optimize(pars_nosigAsb, mm.tools.func_partial(glob_chi2,data=samplesAsb,signal=nosignal),**opts)
none, qsbAb = mm.optimize(pars_sigAb,    mm.tools.func_partial(glob_chi2,data=samplesAb,signal=signal),**opts)

qAsb = (qsbAsb - qbAsb)[0] # extract single sample result
qAb = (qsbAb - qbAb)[0]

#print("qAsb:", qAsb)

do_MC = True
if do_MC:

    # Generate background-only pseudodata to be fitted
    samples0 = joint0.sample(N)
    
    #print("sapmlesAsb:",samplesAsb)
    #print("samples0:", samples0)
    #print("obs_data:", obs_data)
    #quit()
    # Generate signal pseudodata to be fitted
    samples0s = joint0s.sample(N)
        
    print("Fitting w.r.t background-only samples")
    pars_nosig = get_nuis_parameters(analyses,nosignal,samples0)
    pars_sig   = get_nuis_parameters(analyses,signal,samples0)
    fsb = tf.function(mm.tools.func_partial(glob_chi2,data=samples0,signal=nosignal))
    fb  = tf.function(mm.tools.func_partial(glob_chi2,data=samples0,signal=signal))
    pars_nosig, qb = mm.optimize(pars_nosig, fsb, **opts)
    pars_sig, qsb  = mm.optimize(pars_sig,   fb,  **opts)
    q = qsb - qb

    print("Fitting w.r.t signal samples")
    pars_nosig_s  = get_nuis_parameters(analyses,nosignal,samples0s)
    pars_sig_s    = get_nuis_parameters(analyses,signal,samples0s)
    fsb_s = tf.function(mm.tools.func_partial(glob_chi2,data=samples0s,signal=nosignal))
    fb_s  = tf.function(mm.tools.func_partial(glob_chi2,data=samples0s,signal=signal))
    pars_nosig_s, qb_s  = mm.optimize(pars_nosig_s, fsb_s, **opts)
    pars_sig_s, qsb_s   = mm.optimize(pars_sig_s,   fb_s,  **opts)    
    q_s = qsb_s - qb_s


# Fit distributions for observed datasets
print("Fitting w.r.t background-only samples")
pars_nosigO = get_nuis_parameters(analyses,nosignal,obs_data)
pars_sigO   = get_nuis_parameters(analyses,signal,obs_data)
pars_nosigO, qbO = mm.optimize(pars_nosigO, mm.tools.func_partial(glob_chi2,data=obs_data,signal=nosignal),**opts)
pars_sigO, qsbO  = mm.optimize(pars_sigO,   mm.tools.func_partial(glob_chi2,data=obs_data,signal=signal),**opts)
qO = (qsbO - qbO)[0] # extract single sample result
#print("qO:",qO)

nplots = len(s_in)
fig = plt.figure(figsize=(12,4*nplots))
for i in range(nplots):
    ax1 = fig.add_subplot(nplots,2,2*i+1)
    ax2 = fig.add_subplot(nplots,2,2*i+2)
    ax2.set(yscale="log")

    if do_MC:
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

        sns.distplot(qb, color='b', kde=False, ax=ax2, norm_hist=True, label="s={0}".format(s_in[i]))
        sns.distplot(qsb, color='r', kde=False, ax=ax2, norm_hist=True, label="s={0}".format(s_in[i]))

    # Compute and plot asymptotic distributions!

    var_mu_sb = 1./tf.abs(qAsb[i]) 
    var_mu_b  = 1./tf.abs(qAb[i]) 

    #    #var_mu = sign * 1. / LLRA
    #    Eq = LLRA
    #    Varq = sign * 4 * LLRA


    Eq_sb = -1. / var_mu_sb
    Eq_b  = 1. / var_mu_b

    Vq_sb = 4. / var_mu_sb
    Vq_b  = 4. / var_mu_b

    qsbx = Eq_sb + np.linspace(-5*np.sqrt(Vq_sb),5*np.sqrt(Vq_sb),1000)
    qbx  = Eq_b  + np.linspace(-5*np.sqrt(Vq_b), 5*np.sqrt(Vq_b), 1000)

    #qsbx = np.linspace(np.min(qsb),np.max(qsb),1000)
    #qbx  = np.linspace(np.min(qb),np.max(qb),1000)
    qsby = tf.math.exp(tfd.Normal(loc=Eq_sb, scale=tf.sqrt(Vq_sb)).log_prob(qsbx)) 
    qby  = tf.math.exp(tfd.Normal(loc=Eq_b, scale=tf.sqrt(Vq_b)).log_prob(qbx)) 
    
    # Asymptotic p-value and significance for background-only hypothesis test
    apval = tfd.Normal(0,1).cdf((qO[i] - Eq_b) / np.sqrt(Vq_b))
    asig = -tfd.Normal(0,1).quantile(apval)
    
    sns.lineplot(qbx,qby,color='b',ax=ax1)
    sns.lineplot(qsbx,qsby,color='r',ax=ax1)

    sns.lineplot(qbx, qby,color='b',ax=ax2)
    sns.lineplot(qsbx,qsby,color='r',ax=ax2)

    print("qO[{0}]: {1}".format(i,qO[i]))

    ax1.axvline(x=qO.numpy()[i],lw=2,c='k',label="apval={0}, z={1:.1f}".format(apval,asig))
    ax2.axvline(x=qO.numpy()[i],lw=2,c='k',label="apval={0}, z={1:.1f}".format(apval,asig))

    ax1.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
    ax2.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)

plt.tight_layout()
fig.savefig("qsb_dists.png")


