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
analyses_read, SR_map = collider_analyses_from_long_YAML(stream,replace_SR_names=True)
#print(analyses_read[16].name, SR_map["SR220"]); quit()
#analyses = analyses_read[10:] # For testing
#analyses = [analyses_read[16]] # For testing
#analyses = [analyses_read[1]] # For testing
#analyses = [a for a in analyses_read if a.name=="CMS_13TeV_2OSLEP_36invfb"] # For testing
analyses = [a for a in analyses_read if a.name=="CMS_13TeV_MultiLEP_2SSLep_36invfb"]
stream.close()

s_in = [1]
nosignal = {a.name: {'{0}::s'.format(sr): tf.constant([1e-10],dtype=float) for sr in a.SR_names} for a in analyses}
signal = {a.name: {'{0}::s'.format(sr): tf.constant(s_in,dtype=float) for sr in a.SR_names} for a in analyses}
nullnuis = {a.name: {'nuisance': None} for a in analyses} # Use to automatically set nuisance parameters to zero for sample generation

print(nosignal)
print(nullnuis)

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
        pars[a.name] = a.get_nuisance_tensorflow_variables(samples[a.name],signal[a.name])
    return pars

print(deep_merge(nosignal,nullnuis))

joint0  = JMCJoint(analyses,deep_merge(nosignal,nullnuis))
joint0s = JMCJoint(analyses,deep_merge(signal,nullnuis))

samplesAb = joint0.Asamples
samplesAsb = joint0s.Asamples 

#joint0, layouts, counts, samplesAb, distb = get_null_model(nosignal)
#joint0s, layouts, counts, samplesAsb, distsb = get_null_model(signal)

#print("layouts:", layouts)
print("samplesAb:", samplesAb)
#print("samplesAsb:", samplesAsb)
#print("counts:", counts)

print("dict samples?:",joint0.sample(N))
 
# Inspect Asimov samples
# and test pdf evaluation dist by dist

def get_logp(layouts,distlist,samples):
   i=0
   k=0
   logpdf = 0
   all_logp = []
   for layout in layouts:
       for name, v, count in layout:
           d = distlist[k]
           X = samples[i:i+count]
           #print("{0}: {1} = {2}".format(name,v,X))
           logp = d.log_prob(X)
           all_logp += [logp]
           logpdf += logp 
           #print("{0}: {1}: log_pdf = {2}".format(name,v,logp))
           i += count
           k += 1
   return logpdf[0] #, all_logp
 
#print("logpdf combined: ",get_logp(layouts,distb,samplesAsb))

# psbAsb0 = joint0s.log_prob(samplesAsb)
# allp0 = joint0s.log_prob_parts(samplesAsb) 
# psbAsb, allp1 = get_logp(layouts,distsb,samplesAsb)
# print("psbAsb0:", psbAsb0)
# print("psbAsb:", psbAsb)
# #for p0, p1, l in zip(allp0,allp1,layouts[0]):
# #    print("name, p0, p1:",l[0],l[1],p0,p1)
# quit()

#allp0 = joint0s.log_prob_parts(samplesAsb) 
# for p0, l in zip(allp0,layouts[0]):
#     print("name, p0:",l[0],l[1],p0)

# Determine asymptotic test statistic distributions from Asimov samples
qsbAsb = -2*(joint0s.log_prob(samplesAsb))
#qsbAsb_parts = -2*tf.reduce_sum(allp0,axis=0)
qbAsb  = -2*(joint0.log_prob(samplesAsb)) 
#qsbAsb = -2*(get_logp(layouts,distsb,samplesAsb))
#qbAsb  = -2*(get_logp(layouts,distb,samplesAsb)) 
qAsb = qsbAsb - qbAsb
#print("qsbAsb0:", qsbAsb0)
#print("qsbAsb_parts:", qsbAsb_parts)
print("qsbAsb:", qsbAsb)
print("qbAsb:", qbAsb)
print("qAsb:", qAsb)
#quit()

qsbAb = -2*(joint0s.log_prob(samplesAb))
qbAb  = -2*(joint0.log_prob(samplesAb)) 
#qsbAb = -2*(get_logp(layouts,distsb,samplesAb))
#qbAb  = -2*(get_logp(layouts,distb,samplesAb)) 
qAb = qsbAb - qbAb
print("qsbAb:", qsbAb)
print("qbAb:", qbAb)
print("qAb:", qAb)

do_MC = True
if do_MC:

    # Generate background-only pseudodata to be fitted
    samples0 = joint0.sample(N)
    
    # Generate signal pseudodata to be fitted
    samples0s = joint0s.sample(N)
    
    # Define loss function to be minimized
    # Use @tf.function decorator to compile target function into graph mode for faster evaluation
    # Takes a while to compile the graph at startup, but execution is quite a bit faster
    # Turn off for rapid testing, turn on for production.
    @tf.function
    def glob_chi2(pars,data,signal=signal):
        print("pars:",pars)
        print("signal:",signal)
        joint_s = JMCJoint(analyses,deep_merge(pars,signal))
        q = -2*joint_s.log_prob(data)
        print("q:",q)
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
    pars_s = get_nuis_parameters(analyses,signal,samples0s)
    pars_s, qsb_s = mm.optimize(pars_s,mm.tools.func_partial(glob_chi2,data=samples0s,signal=signal),**opts)
    pars = get_nuis_parameters(analyses,signal,samples0)
    pars, qsb = mm.optimize(pars,mm.tools.func_partial(glob_chi2,data=samples0,signal=signal),**opts)
    
    print("Fitting background-only hypothesis")
    parsb = get_nuis_parameters(analyses,nosignal,samples0)
    parsb, qb = mm.optimize(parsb,mm.tools.func_partial(glob_chi2,data=samples0,signal=nosignal),**opts)
    parsb_s = get_nuis_parameters(analyses,nosignal,samples0s)
    parsb_s, qb_s = mm.optimize(parsb_s,mm.tools.func_partial(glob_chi2,data=samples0s,signal=nosignal),**opts)
    
    q = qsb - qb
    q_s = qsb_s - qb_s

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
    var_mu_sb = 1/tf.abs(qAsb[i]) 
    var_mu_b  = 1/tf.abs(qAb[i]) 

    Eq_sb = -1 / var_mu_sb
    Eq_b  = 1 / var_mu_b

    Vq_sb = 4 / var_mu_sb
    Vq_b  = 4 / var_mu_b

    qsbx = Eq_sb + np.linspace(-5*np.sqrt(Vq_sb),5*np.sqrt(Vq_sb),1000)
    qbx  = Eq_b  + np.linspace(-5*np.sqrt(Vq_b), 5*np.sqrt(Vq_b), 1000)

    #qsbx = np.linspace(np.min(qsb),np.max(qsb),1000)
    #qbx  = np.linspace(np.min(qb),np.max(qb),1000)
    qsby = tf.math.exp(tfd.Normal(loc=Eq_sb, scale=tf.sqrt(Vq_sb)).log_prob(qsbx)) 
    qby  = tf.math.exp(tfd.Normal(loc=Eq_b, scale=tf.sqrt(Vq_b)).log_prob(qbx)) 

    sns.lineplot(qbx,qby,color='b',ax=ax1)
    sns.lineplot(qsbx,qsby,color='r',ax=ax1)

    sns.lineplot(qbx, qby,color='b',ax=ax2)
    sns.lineplot(qsbx,qsby,color='r',ax=ax2)

    ax1.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
    ax2.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)

plt.tight_layout()
fig.savefig("qsb_dists.png")


