from analysis import collider_analyses_from_long_YAML, JMCJoint, deep_merge
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as sps

N = int(5e2)
do_mu_tests=True
do_gof_tests=True

print("Starting...")

# Load analysis metadata from YAML and construct helper ColliderAnalysis objects for it
f = 'old_junk/CBit_analyses.yaml'
stream = open(f, 'r')
analyses_read, SR_map, iSR_map = collider_analyses_from_long_YAML(stream,replace_SR_names=True)
#print(analyses_read[16].name, SR_map["SR220"]); quit()
#analyses = analyses_read[10:] # For testing
#analyses = [analyses_read[16]] # For testing
#analyses = analyses_read[2:4] # For testing
#analyses_read = {name: a for name,a in analyses_read.items() if a.name=="CMS_13TeV_2OSLEP_36invfb"}
#analyses = [a for a in analyses_read if a.name=="CMS_13TeV_MultiLEP_2SSLep_36invfb"]
#analyses = [a for a in analyses_read if a.name=="ATLAS_13TeV_3b_discoverySR_24invfb"]
analyses_read = {name: a for name,a in analyses_read.items() if a.name=="TEST"}
#analyses = [a for a in analyses_read if a.name=="ATLAS_13TeV_RJ3L_3Lep_36invfb"]
#analyses = [a for a in analyses_read if a.name=="CMS_13TeV_2LEPsoft_36invfb"]
#analyses_read = {name: a for name,a in analyses_read.items() if a.name=="CMS_8TeV_MultiLEP_3Lep_20invfb"}
#analyses = analyses_read
stream.close()

#s_in = [0.2,.5,1.,2.]
#s_in = [0.1,1.,10.,20.]
s_in = [1.,10.,20.,50.]
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


# mu=1 vs mu=0 tests
if do_mu_tests:
    for a in analyses_read.values():
        print("Simulating analysis {0}".format(a.name))
        analyses = {a.name: a}
    
        # Create joint distributions
        joint   = JMCJoint(analyses) # Version for fitting (object is left with fitted parameters upon fitting)
        joint0  = JMCJoint(analyses,deep_merge(nosignal,nullnuis))
        joint0s = JMCJoint(analyses,deep_merge(signal,nullnuis))
        
        # Get Asimov samples for nuisance MLEs with fixed signal hypotheses
        samplesAb = joint0.Asamples
        samplesAsb = joint0s.Asamples 
        #print("sapmlesAsb:",samplesAsb)
        #print("sapmlesAb:",samplesAb)
 
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
        qbAsb, joint_fitted, pars = joint.fit_nuisance(nosignal, samplesAsb,log_tag='bAsb')
        qsbAb, joint_fitted, pars = joint.fit_nuisance(signal, samplesAb,log_tag='sbAsb')
        print("qbAsb:", qbAsb)
        print("qsbAb:", qsbAb)
  
        qAsb = (qsbAsb - qbAsb)[0] # extract single sample result
        qAb = (qsbAb - qbAb)[0]
        
        #print("qsbAb:", qbAsb)
        # Check that the fitting worked
        #print("qsbAb_fit:", -2*joint.log_prob(samplesAb))
        #quit()

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
        
            if do_gof_tests:
                print("Fitting GOF w.r.t background-only samples")
                qgof_b, joint_gof_fitted_b, gof_pars_b  = joint.fit_all(samples0, log_tag='gof_all_b')
                print("Fitting GOF w.r.t signal samples")
                qgof_sb, joint_gof_fitted_sb ,gof_pars_sb  = joint.fit_all(samples0s, log_tag='gof_all_sb')

                # Obtain function to compute neg2logl for fitted samples, for any fixed input signal,
                # with nuisance parameters analytically profiled out using a second order Taylor expansion
                # about the GOF best fit points.
                #print("fitted pars:", gof_pars_b)
                f = joint_gof_fitted_b.quad_loglike_f(samples0)
                qsb_quad = f(signal)
                qb_quad = f(nosignal) # Only one of these so can easily do it numerically, but might be more consistent to use same approx. for both.
                #print("qsb_quad:", qsb_quad)
                q_quad = qsb_quad - qb_quad

                print("Fitting w.r.t background-only samples")
                qb , joint_fitted, nuis_pars_b = joint.fit_nuisance(nosignal, samples0, log_tag='qb')
                qsb, joint_fitted, nuis_pars_s = joint.fit_nuisance(signal, samples0, log_tag='qsb')
                #print("qsb:", qsb)
                #print("nuis_s:", joint.descale_pars(nuis_pars_s))
                q = qsb - qb # mu=0 distribution
                print("Fitting w.r.t signal samples")
                qb_s , joint_fitted, nuis_pars = joint.fit_nuisance(nosignal, samples0s)
                qsb_s, joint_fitted, nuis_pars = joint.fit_nuisance(signal, samples0s)
                q_s = qsb_s - qb_s #mu=1 distribution
        
                # GOF distributions
                qgofb_true = qb - qgof_b # Test of b-only when b is true
                qgofsb_true = qsb_s - qgof_sb # Test of s when s is true
                # the above should both be asymptotically chi^2
                N_gof_pars = sum([par.shape[-1] for a in gof_pars_b.values() for par in a.values()])
                N_nuis_pars = sum([par.shape[-1] for a in nuis_pars.values() for par in a.values()])
                #print("N_gof_pars:",N_gof_pars)
                #print("N_nuis_pars:")
                DOF = N_gof_pars - N_nuis_pars

        # Fit distributions for observed datasets
        print("Fitting w.r.t observed data")
        qbO , joint_fitted, pars = joint.fit_nuisance(nosignal, obs_data)
        qsbO, joint_fitted, pars = joint.fit_nuisance(signal, obs_data)
        qO = (qsbO - qbO)[0] # extract single sample result

        print("Fitting GOF w.r.t observed data")
        qgof_obs, joint_fitted, pars  = joint.fit_all(obs_data, log_tag='gof_obs')
        qgofOb  = qbO - qgof_obs
        qgofOsb = qsbO - qgof_obs
        
        print("GOF BF pars:", pars)

        nplots = len(s_in)
        fig  = plt.figure(figsize=(12,4*nplots))
        fig2 = plt.figure(figsize=(12,4*nplots))
        for i in range(nplots):
            ax1 = fig.add_subplot(nplots,2,2*i+1)
            ax2 = fig.add_subplot(nplots,2,2*i+2)
            ax2.set(yscale="log")

            # GOF plots
            ax3 = fig2.add_subplot(nplots,2,2*i+1)
            ax4 = fig2.add_subplot(nplots,2,2*i+2)
            ax3.set(yscale="log")
         
            if do_MC:
                qb  = q[:,i].numpy()
                qb_quad = q_quad[:,i].numpy()
                qsb = q_s[:,i].numpy()
                if np.sum(np.isfinite(qb)) < 2:
                    print("qb mostly nan!")
                if np.sum(np.isfinite(qsb)) < 2:
                    print("qsb mostly nan!")
                qb = qb[np.isfinite(qb)]
                qsb = qsb[np.isfinite(qsb)]
        
                sns.distplot(qb , bins=50, color='b',kde=False, ax=ax1, norm_hist=True, label="s={0}".format(s_in[i]))
                sns.distplot(qb_quad , bins=50, color='m',kde=False, ax=ax1, norm_hist=True, label="s={0}".format(s_in[i]))
                sns.distplot(qsb, bins=50, color='r', kde=False, ax=ax1, norm_hist=True, label="s={0}".format(s_in[i]))
        
                sns.distplot(qb, color='b', kde=False, ax=ax2, norm_hist=True, label="s={0}".format(s_in[i]))
                sns.distplot(qb_quad , bins=50, color='m',kde=False, ax=ax2, norm_hist=True, label="s={0}".format(s_in[i]))
                sns.distplot(qsb, color='r', kde=False, ax=ax2, norm_hist=True, label="s={0}".format(s_in[i]))

                # GOF
                if do_gof_tests:
                    sns.distplot(qgofb_true      , color='b', kde=False, ax=ax3, norm_hist=True, label="s={0}".format(s_in[i]))
                    sns.distplot(qgofsb_true[:,i], color='r', kde=False, ax=ax3, norm_hist=True, label="s={0}".format(s_in[i]))
                    sns.distplot(qgofb_true      , color='b', kde=False, ax=ax4, norm_hist=True, label="s={0}".format(s_in[i]))
                    sns.distplot(qgofsb_true[:,i], color='r', kde=False, ax=ax4, norm_hist=True, label="s={0}".format(s_in[i]))
          
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
        
            if do_gof_tests:
                #qx = np.linspace(0, tfd.Chi2(df=DOF).quantile(tfd.Normal(0,1).cdf(6.)),1000) # TODO: apparantly Quantile not implemented yet for Chi2 in tf
                qx = np.linspace(0, sps.chi2(df=DOF).ppf(tfd.Normal(0,1).cdf(5.)),1000) # 6 sigma too far for tf, cdf is 1. single-precision float I guess
                qy = tf.math.exp(tfd.Chi2(df=DOF).log_prob(qx))
                sns.lineplot(qx,qy,color='g',ax=ax3)
                sns.lineplot(qx,qy,color='g',ax=ax4)

                ax3.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
                ax4.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
   
        fig.tight_layout()
        fig.savefig("qsb_dists_{0}.png".format(a.name))

        if do_gof_tests:
            fig2.tight_layout()
            fig2.savefig("gof_dists_{0}.png".format(a.name))
