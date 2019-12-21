from analysis import collider_analyses_from_long_YAML, JMCJoint, deep_merge
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as sps
import scipy.interpolate as spi

N = int(5e3)
do_mu_tests=True
do_gof_tests=True

def eCDF(x):
    """Get empirical CDF of some samples"""
    return np.arange(1, len(x)+1)/float(len(x))

def CDFf(samples,reverse=False):
    """Return interpolating function for CDF of some simulated samples"""
    if reverse:
        s = np.argsort(samples[np.isfinite(samples)],axis=0)[::-1] 
    else:
        s = np.argsort(samples[np.isfinite(samples)],axis=0)
    ecdf = eCDF(samples[s])
    CDF = spi.interp1d([-1e99]+list(samples[s])+[1e99],[ecdf[0]]+list(ecdf)+[ecdf[1]])
    return CDF, s #pvalue may be 1 - CDF(obs), depending on definition/ordering

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
#analyses_read = {name: a for name,a in analyses_read.items() if a.name=="ATLAS_13TeV_RJ3L_3Lep_36invfb"}
#analyses = [a for a in analyses_read if a.name=="CMS_13TeV_2LEPsoft_36invfb"]
#analyses_read = {name: a for name,a in analyses_read.items() if a.name=="CMS_8TeV_MultiLEP_3Lep_20invfb"}
#analyses = analyses_read
stream.close()

#s_in = [0.2,.5,1.,2.,5.]
#s_in = [0.1,1.,10.,20.]
s_in = [1.,10.,20.,50.]
nosignal = {a.name: {'s': tf.constant([[0. for sr in a.SR_names]],dtype=float)} for a in analyses_read.values()}
signal = {a.name: {'s': tf.constant([[s for sr in a.SR_names] for s in s_in], dtype=float)} for a in analyses_read.values()}
nullnuis = {a.name: {'nuisance': None} for a in analyses_read.values()} # Use to automatically set nuisance parameters to zero for sample generation
Ns = len(s_in)

# Generate grid of samples for TEST analysis
def ndim_grid(start,stop,N):
    # Set number of dimensions
    ndims = len(start)

    # List of ranges across all dimensions
    L = [np.linspace(start[i],stop[i],N) for i in range(ndims)]

    # Finally use meshgrid to form all combinations corresponding to all 
    # dimensions and stack them as M x ndims array
    return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T

start = []
stop = []
a = analyses_read["TEST"]
b     = a.SR_b 
b_sys = a.SR_b_sys 
for bi,bsysi in zip(b,b_sys):
    start += [-4*bsysi]
    stop  += [+4*bsysi]
#sgrid = ndim_grid(start,stop,20)
sigs = np.linspace(-4*bsysi,+4*bsysi,50)
print("sigs:", sigs)
#np.random.shuffle(sigs)
sgrid = tf.expand_dims(tf.constant(sigs,dtype=float),axis=1)
#signal = {"TEST": {'s': tf.constant(sgrid,dtype=float)}}
#Ns = len(sgrid)

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

                print("Fitting w.r.t background-only samples")
                qb , joint_fitted_b, nuis_pars_b = joint.fit_nuisance(nosignal, samples0, log_tag='qb')
                qsb, joint_fitted_sb, nuis_pars_s = joint.fit_nuisance(signal, samples0, log_tag='qsb')
                #print("qsb:", qsb)
                #print("nuis_s:", joint.descale_pars(nuis_pars_s))
                q = qsb - qb # mu=0 distribution

                #Experimental: Fit mu scaling test statistic
                print("Fitting scale w.r.t background-only samples")
                qmub, joint_fitted_mub, pars_mub = joint.fit_nuisance_and_scale(signal,samples0,log_tag='qmub')
                qmu0 = qb - qmub

                # Obtain function to compute neg2logl for fitted samples, for any fixed input signal,
                # with nuisance parameters analytically profiled out using a second order Taylor expansion
                # about the GOF best fit points.
                #print("fitted pars:", gof_pars_b)
                f1 = joint_gof_fitted_b.quad_loglike_f(samples0)

                # What if we expand about the no-signal point instead? Just to see...
                f2 = joint_fitted_b.quad_loglike_f(samples0)
                # Huh, seems to work way better. I guess it should be better when the test signals are small?

                # Can we combine the two? Weight be inverse square euclidean distance in GOF parameter space?
                # Something smarter?
                def combf(signal):
                    p1 = nosignal
                    p2 = gof_pars_b
                    print("p1:", p1)
                    print("p2:", p2)
                    print("signal", signal)
                    dist1_squared = 0
                    dist2_squared = 0
                    for ka,a in signal.items():
                        for pa,p in a.items():
                            dist1_squared += (tf.expand_dims(signal[ka][pa],axis=0) - tf.expand_dims(p1[ka][pa],axis=0))**2 
                            dist2_squared += (tf.expand_dims(signal[ka][pa],axis=0) - p2[ka][pa])**2 
                    w1 = tf.reduce_sum(1./dist1_squared,axis=-1)
                    w2 = tf.reduce_sum(1./dist2_squared,axis=-1)
                    print("w1:", w1)
                    print("w2:", w2)
                    w = w1+w2
                    return (w1/w)*f1(signal) + (w2/w)*f2(signal)

                qsb_quad = f2(signal)
                #qsb_quad = combf(signal)
                #qb_quad = f2(nosignal) # Only one of these so can easily do it numerically, but might be more consistent to use same approx. for both.
                #print("qsb_quad:", qsb_quad)
                #q_quad = qsb_quad - qb_quad
                q_quad = qsb_quad - qb # Using quad approx only for signal half. Biased, but maybe better p-value behaviour.

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

        nplots = Ns
        fig  = plt.figure(figsize=(12,4*nplots))
        fig2 = plt.figure(figsize=(12,4*nplots))
        fig3 = plt.figure(figsize=(6,4*nplots))
        fig4 = plt.figure(figsize=(12,4*nplots))
        for i in range(nplots):
            ax1 = fig.add_subplot(nplots,2,2*i+1)
            ax2 = fig.add_subplot(nplots,2,2*i+2)
            ax2.set(yscale="log")

            # GOF plots
            ax3 = fig2.add_subplot(nplots,2,2*i+1)
            ax4 = fig2.add_subplot(nplots,2,2*i+2)
            ax3.set(yscale="log")
       
            # qmu plots
            ax41 = fig4.add_subplot(nplots,2,2*i+1)
            ax42 = fig4.add_subplot(nplots,2,2*i+2)
            ax42.set(yscale="log")

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
        
                sns.distplot(qb , bins=50, color='b',kde=False, ax=ax1, norm_hist=True)
                sns.distplot(qb_quad , bins=50, color='m',kde=False, ax=ax1, norm_hist=True)
                sns.distplot(qsb, bins=50, color='r', kde=False, ax=ax1, norm_hist=True)
        
                sns.distplot(qb, color='b', kde=False, ax=ax2, norm_hist=True)
                sns.distplot(qb_quad , bins=50, color='m',kde=False, ax=ax2, norm_hist=True)
                sns.distplot(qsb, color='r', kde=False, ax=ax2, norm_hist=True)

                # GOF
                if do_gof_tests:
                    sns.distplot(qgofb_true      , color='b', kde=False, ax=ax3, norm_hist=True)
                    sns.distplot(qgofsb_true[:,i], color='r', kde=False, ax=ax3, norm_hist=True)
                    sns.distplot(qgofb_true      , color='b', kde=False, ax=ax4, norm_hist=True)
                    sns.distplot(qgofsb_true[:,i], color='r', kde=False, ax=ax4, norm_hist=True)

                # qmu
                sns.distplot(qmu0[:,i], color='b', kde=False, ax=ax41, norm_hist=True)
                sns.distplot(qmu0[:,i], color='b', kde=False, ax=ax42, norm_hist=True)
           
                # Quad vs MC local p-value comparison
                # Compute quad p-values by looking up simulated q-values on the MC distribution.
                # Though in principle could construct test based directly on quad values. Legit
                # frequentist test if we decide on this procedure in advance and can MC the distribution?
                # Probably just has a bit less power or something.
                # Anyway still good to know how similar results are to the full MC result.
                MCcdf, order = CDFf(qb)
                print("qb:",qb)
                print("MCcdf(qb):", MCcdf(qb))
                fullMCp = -sps.norm.ppf(MCcdf(qb))
                quadMCp = -sps.norm.ppf(MCcdf(qb_quad))

                ax31 = fig3.add_subplot(nplots,1,i+1)
                diag = np.min(fullMCp), np.max(fullMCp)
                print("diag:",diag)
                ax31.plot(diag,diag,c='k')
                ax31.scatter(fullMCp,quadMCp,s=1,c='b')
                ax31.set_xlabel("full MC sigma")
                ax31.set_ylabel("quad. approx sigma")

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

            qx = np.linspace(0, sps.chi2(df=1).ppf(tfd.Normal(0,1).cdf(5.)),1000) # 6 sigma too far for tf, cdf is 1. single-precision float I guess
            qy = tf.math.exp(tfd.Chi2(df=1).log_prob(qx))
            sns.lineplot(qx,qy,color='b',ax=ax41)
            sns.lineplot(qx,qy,color='b',ax=ax42)

   
        fig.tight_layout()
        fig.savefig("qsb_dists_{0}.png".format(a.name))

        fig3.tight_layout()
        fig3.savefig("qb_vs_qquad{0}.png".format(a.name))

        if do_gof_tests:
            fig2.tight_layout()
            fig2.savefig("gof_dists_{0}.png".format(a.name))

        fig4.tight_layout()
        fig4.savefig("qmu0_dists_{0}.png".format(a.name))

