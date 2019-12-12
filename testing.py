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
#analyses_read = {name: a for name,a in analyses_read.items() if a.name=="TEST"}
#analyses = [a for a in analyses_read if a.name=="ATLAS_13TeV_RJ3L_3Lep_36invfb"]
#analyses = [a for a in analyses_read if a.name=="CMS_13TeV_2LEPsoft_36invfb"]
analyses_read = {name: a for name,a in analyses_read.items() if a.name=="CMS_8TeV_MultiLEP_3Lep_20invfb"}
#analyses = analyses_read
stream.close()

s_in = [.5,1.]
#s_in = [1.,10.]
nosignal = {a.name: {'s': tf.constant([[0. for sr in a.SR_names]],dtype=float)} for a in analyses_read.values()}
signal = {a.name: {'s': tf.constant([[s for sr in a.SR_names] for s in s_in], dtype=float)} for a in analyses_read.values()}
nullnuis = {a.name: {'nuisance': None} for a in analyses_read.values()} # Use to automatically set nuisance parameters to zero for sample generation

# ATLAS_13TeV_RJ3L_3Lep_36invfb
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

def glob_chi2(pars,analyses,data,pre_scaled_pars):
        joint_s = JMCJoint(analyses,pars,pre_scaled_pars)
        q = -2*joint_s.log_prob(data)
        total_loss = tf.math.reduce_sum(q)
        return total_loss, None, q

def optimize(pars,*args,**kwargs):
    """Wrapper for optimizer step that skips it if the initial guesses are known
       to be exact MLEs"""

    opts = {"optimizer": "Adam",
            "step": 0.3,
            "tol": 0.01,
            "grad_tol": 1e-4,
            "max_it": 100,
            "max_same": 5
            }

    exact_MLEs = True
    for a in analyses.values():
        if a.cov is not None: exact_MLEs = False
    if exact_MLEs:
        print("All starting MLE guesses are exact: skipping optimisation") 
        loss, none, q = glob_chi2(pars,*args,**kwargs)
    else:
        print("Beginning optimisation")
        f = tf.function(mm.tools.func_partial(glob_chi2,*args,**kwargs))
        none, q = mm.optimize(pars, f, **opts)
    return q

# mu=1 vs mu=0 tests
do_mu_tests=True
if do_mu_tests:
    for a in analyses_read.values():
        print("Simulating analysis {0}".format(a.name))
        analyses = {a.name: a}
    
        # Create joint distributions
        joint0  = JMCJoint(analyses,deep_merge(nosignal,nullnuis))
        joint0s = JMCJoint(analyses,deep_merge(signal,nullnuis))
        
        # Get Asimov samples for nuisance MLEs with fixed signal hypotheses
        samplesAb = joint0.Asamples
        samplesAsb = joint0s.Asamples 
        print("sapmlesAsb:",samplesAsb)
        print("sapmlesAb:",samplesAb)
 
        # Get observed data
        obs_data = joint0.Osamples
        
        # Evaluate distributions for Asimov datasets, in case where we
        # know that the MLEs for those samples are the true parameters
        qsbAsb = -2*(joint0s.log_prob(samplesAsb))
        qbAb  = -2*(joint0.log_prob(samplesAb))
        print("qsbAsb:", qsbAsb)
        print("qbAb:", qbAb)
        
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
        pars_nosigAsb = joint0.get_nuis_parameters(nosignal,samplesAsb)
        pars_sigAb    = joint0.get_nuis_parameters(signal,samplesAb)
        #none, qbAsb = mm.optimize(pars_nosigAsb, mm.tools.func_partial(glob_chi2,data=samplesAsb,signal=nosignal),**opts)
        #none, qsbAb = mm.optimize(pars_sigAb,    mm.tools.func_partial(glob_chi2,data=samplesAb,signal=signal),**opts)
        qbAsb = optimize(deep_merge(pars_nosigAsb,nosignal),analyses,samplesAsb,pre_scaled_pars='nuis')
        qsbAb = optimize(deep_merge(pars_sigAb   ,signal),  analyses,samplesAb ,pre_scaled_pars='nuis')
    
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
            pars_nosig = joint0.get_nuis_parameters(nosignal,samples0)
            pars_sig   = joint0.get_nuis_parameters(signal,samples0)
            # fsb = tf.function(mm.tools.func_partial(glob_chi2,data=samples0,signal=nosignal))
            # fb  = tf.function(mm.tools.func_partial(glob_chi2,data=samples0,signal=signal))
            # pars_nosig, qb = mm.optimize(pars_nosig, fsb, **opts) ?? are these right?
            # pars_sig, qsb  = mm.optimize(pars_sig,   fb,  **opts) ??
            qb  = optimize(deep_merge(pars_nosig,nosignal),analyses,samples0,pre_scaled_pars='nuis')
            qsb = optimize(deep_merge(pars_sig  ,signal),  analyses,samples0,pre_scaled_pars='nuis')
    
            q = qsb - qb
        
            print("Fitting w.r.t signal samples")
            pars_nosig_s  = joint0.get_nuis_parameters(nosignal,samples0s)
            pars_sig_s    = joint0.get_nuis_parameters(signal,samples0s)
            # fsb_s = tf.function(mm.tools.func_partial(glob_chi2,data=samples0s,signal=nosignal))
            # fb_s  = tf.function(mm.tools.func_partial(glob_chi2,data=samples0s,signal=signal))
            # pars_nosig_s, qb_s  = mm.optimize(pars_nosig_s, fsb_s, **opts)
            # pars_sig_s, qsb_s   = mm.optimize(pars_sig_s,   fb_s,  **opts)    
            qb_s  = optimize(deep_merge(pars_nosig_s,nosignal),analyses,samples0s,pre_scaled_pars='nuis')
            qsb_s = optimize(deep_merge(pars_sig_s  ,signal),  analyses,samples0s,pre_scaled_pars='nuis')
    
            q_s = qsb_s - qb_s
        
        
        # Fit distributions for observed datasets
        print("Fitting w.r.t background-only samples")
        pars_nosigO = joint0.get_nuis_parameters(nosignal,obs_data)
        pars_sigO   = joint0.get_nuis_parameters(signal,obs_data)
        #pars_nosigO, qbO = mm.optimize(pars_nosigO, mm.tools.func_partial(glob_chi2,data=obs_data,signal=nosignal),**opts)
        #pars_sigO, qsbO  = mm.optimize(pars_sigO,   mm.tools.func_partial(glob_chi2,data=obs_data,signal=signal),**opts)
        qbO  = optimize(deep_merge(pars_nosigO,nosignal),analyses,obs_data,pre_scaled_pars='nuis')
        qsbO = optimize(deep_merge(pars_sigO  ,signal),  analyses,obs_data,pre_scaled_pars='nuis')
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
            apval = tfd.Normal(0,1).cdf(np.abs(qO[i] - Eq_b) / np.sqrt(Vq_b))
            asig = -tfd.Normal(0,1).quantile(apval)
            
            sns.lineplot(qbx,qby,color='b',ax=ax1)
            sns.lineplot(qsbx,qsby,color='r',ax=ax1)
        
            sns.lineplot(qbx, qby,color='b',ax=ax2)
            sns.lineplot(qsbx,qsby,color='r',ax=ax2)
        
            #print("qO[{0}]: {1}".format(i,qO[i]))
        
            ax1.axvline(x=qO.numpy()[i],lw=2,c='k',label="apval={0}, z={1:.1f}".format(apval,asig))
            ax2.axvline(x=qO.numpy()[i],lw=2,c='k',label="apval={0}, z={1:.1f}".format(apval,asig))
        
            ax1.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
            ax2.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
        
        plt.tight_layout()
        fig.savefig("qsb_dists_{0}.png".format(a.name))

# GOF tests
do_gof_tests=False
if do_gof_tests:
    print("Performing GOF tests")
    for a in analyses_read.values():
        print("Simulating analysis {0}".format(a.name))
        analyses = [a]
    
        # Create joint distributions
        joint0  = JMCJoint(analyses,deep_merge(nosignal,nullnuis))
        joint0s = JMCJoint(analyses,deep_merge(signal,nullnuis))

        do_MC = True
        if do_MC:
        
            # Generate background-only pseudodata to be fitted
            samples0 = joint0.sample(N)
            
            # Generate signal pseudodata to be fitted
            samples0s = joint0s.sample(N)
                
            print("Fitting w.r.t background-only samples")
            pars_nosig = joint0.get_all_parameters(samples0)
            pars_sig   = joint0.get_all_parameters(samples0)
            qb  = optimize(pars_nosig,analyses,samples0,nosignal,pre_scaled_pars='all')
            qsb = optimize(pars_sig,  analyses,samples0,signal,pre_scaled_pars='all')
            pars_nosig = joint0.get_nuis_parameters(nosignal,samples0)
            pars_sig   = joint0.get_nuis_parameters(signal,samples0)
            qb  = optimize(pars_nosig,analyses,samples0,nosignal,pre_scaled_pars='all')
            qsb = optimize(pars_sig,  analyses,samples0,signal,pre_scaled_pars='all')
     
            q = qsb - qb
        
            print("Fitting w.r.t signal samples")
            pars_nosig_s  = joint0.get_nuis_parameters(nosignal,samples0s)
            pars_sig_s    = joint0.get_nuis_parameters(signal,samples0s)
            # fsb_s = tf.function(mm.tools.func_partial(glob_chi2,data=samples0s,signal=nosignal))
            # fb_s  = tf.function(mm.tools.func_partial(glob_chi2,data=samples0s,signal=signal))
            # pars_nosig_s, qb_s  = mm.optimize(pars_nosig_s, fsb_s, **opts)
            # pars_sig_s, qsb_s   = mm.optimize(pars_sig_s,   fb_s,  **opts)    
            qb_s  = optimize(pars_nosig_s,analyses,samples0s,nosignal,pre_scaled_pars='all')
            qsb_s = optimize(pars_sig_s,  analyses,samples0s,signal,pre_scaled_pars='all')
    
            q_s = qsb_s - qb_s
        
        
        # Fit distributions for observed datasets
        print("Fitting w.r.t background-only samples")
        pars_nosigO = joint0.get_nuis_parameters(nosignal,obs_data)
        pars_sigO   = joint0.get_nuis_parameters(signal,obs_data)
        #pars_nosigO, qbO = mm.optimize(pars_nosigO, mm.tools.func_partial(glob_chi2,data=obs_data,signal=nosignal),**opts)
        #pars_sigO, qsbO  = mm.optimize(pars_sigO,   mm.tools.func_partial(glob_chi2,data=obs_data,signal=signal),**opts)
        qbO  = optimize(pars_nosigO,analyses,obs_data,nosignal)
        qsbO = optimize(pars_sigO,analyses,obs_data,signal)
        qO = (qsbO - qbO)[0] # extract single sample result
 
