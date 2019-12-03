from analysis import collider_analyses_from_long_YAML
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import massminimize as mm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

N = int(1e3)

print("Starting...")

# Load analysis metadata from YAML and construct helper ColliderAnalysis objects for it
f = 'old_junk/CBit_analyses.yaml'
stream = open(f, 'r')
analyses_read, SR_map = collider_analyses_from_long_YAML(stream,replace_SR_names=True)
#print(analyses_read[16].name, SR_map["SR220"]); quit()
#analyses_read = analyses_read[10:] # For testing
#analyses_read = [analyses_read[16]] # For testing
analyses_read = [analyses_read[1]] # For testing
#analyses_read = [a for a in analyses_read if a.name=="CMS_13TeV_2OSLEP_36invfb"] # For testing
stream.close()

s_in = [1]
nosignal = {a.name: {sr: tf.constant([1e-10],dtype=float) for sr in a.SR_names} for a in analyses_read}
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
    return tfd.JointDistributionSequential(null_dists), layouts, counts, Asamples, null_dists

def get_free_model(pars,signal):
    free_dists = []
    layouts = []
    counts = []
    for a in analyses_read:
        d, layout, count = a.tensorflow_free_model(signal[a.name],pars[a.name])
        free_dists += d
        layouts += [layout]
        counts += [count]
    return tfd.JointDistributionSequential(free_dists), layouts, counts, free_dists

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

joint0, layouts, counts, samplesAb, distb = get_null_model(nosignal)
joint0s, layouts, counts, samplesAsb, distsb = get_null_model(signal)

#print("layouts:", layouts)
#print("samplesAb:", samplesAb)
#print("samplesAsb:", samplesAsb)
#print("counts:", counts)

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
    #@tf.function
    def glob_chi2(pars,data,signal=signal):
        joint_s, layouts, counts, dists = get_free_model(pars,signal)
        q = -2*joint_s.log_prob(data)
        #q = -2*get_logp(layouts,dists,data)
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
    pars_s = get_free_parameters(samples0s,layouts,counts,signal)
    pars_s, qsb_s = mm.optimize(pars_s,mm.tools.func_partial(glob_chi2,data=samples0s,signal=signal),**opts)
    pars = get_free_parameters(samples0,layouts,counts,signal)
    pars, qsb = mm.optimize(pars,mm.tools.func_partial(glob_chi2,data=samples0,signal=signal),**opts)
    
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


