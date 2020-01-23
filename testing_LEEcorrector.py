"""Testing LEEcorrector objects"""

from analysis import collider_analyses_from_long_YAML, LEECorrectorMaster
from common import deep_merge, CDFf
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as sps
import scipy.interpolate as spi

N = int(1e4)
do_mu_tests=True
do_gof_tests=True


print("Starting...")

# Load analysis metadata from YAML and construct helper ColliderAnalysis objects for it
f = 'old_junk/test.yaml'
stream = open(f, 'r')
analyses_read, SR_map, iSR_map = collider_analyses_from_long_YAML(stream,replace_SR_names=True)

# Generate grid of numbers in N dimensions, as list of coordinates
def ndim_grid(start,stop,N):
    # Set number of dimensions
    ndims = len(start)

    # List of ranges across all dimensions
    L = [np.linspace(start[i],stop[i],N) for i in range(ndims)]

    # Finally use meshgrid to form all combinations corresponding to all 
    # dimensions and stack them as M x ndims array
    return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T

def get_grid(analyses,N):
    """Generate simple grid over simple signal model for all SRs in 'analyses'"""
    start = []
    stop = []
    size = []
    for a in analyses.values():
        b     = a.SR_b 
        b_sys = a.SR_b_sys
        size += [len(b)]
        for bi,bsysi in zip(b,b_sys):
            start += [-4*bsysi]
            stop  += [+4*bsysi]
    sgrid = ndim_grid(start,stop,N)
    signal = {}
    i = 0
    for a,n in zip(analyses.values(),size):
        signal[a.name] = {'s': tf.constant(sgrid[:,i:i+n],dtype=float)}
        i += n
    Ns = len(sgrid)
    return Ns, signal

Ns, signals = get_grid(analyses_read,20) # For testing only! Will die if used for more than e.g. 3 total SRs.

class SigGen:
    """Object to supply signal hypotheses in chunks
       Replace with something that e.g. reads from HDF5 file
       in real cases."""
    def __init__(self,Ns,signals):
        self.count = Ns
        self.chunk_size = 1000
        self.signals = signals
        self.j = 0

    def reset(self): 
        self.j = 0

    def next(self):
        j = self.j
        size = self.chunk_size
        chunk = {name: {par: dat[j:j+size] for par,dat in a.items()} for name,a in self.signals.items()}
        self.j += size
        return chunk, list(range(j,j+size))

nosignal = {a.name: {'s': tf.constant([[0. for sr in a.SR_names]],dtype=float)} for a in analyses_read.values()}
DOF = 3

# Specific signal whose local distributions and p-values we would like to know
# This case is 'cherry-picked' to match the "observed" counts in every signal region of every analysis
signal_test = {a.name: {'s': tf.constant([[n for n in a.SR_n]], dtype=float)} for a in analyses_read.values()}

path = 'TEST'
master_name = 'all'
nullname = 'background'
lee = LEECorrectorMaster(analyses_read,path,master_name,nosignal,nullname)
#lee.ensure_equal_events()
#lee.add_events(int(1e3))
#lee.process_null()
sig_name = "cherry_picked"
#lee.process_signal_local(signal_test,name=sig_name)
#lee.process_signals(SigGen(Ns,signals),new_events_only=True,event_batch_size=10000,dbtype='hdf5')
#lee.process_signals(SigGen(Ns,signals),new_events_only=True,event_batch_size=10000,dbtype='sqlite')

bootstrap_neg2logL, bootstrap_b_neg2logL = lee.get_bootstrap_sample(1000,batch_size=100)
#bootstrap_neg2logL, bootstrap_b_neg2logL = lee.get_bootstrap_sample('all',batch_size=100,dbtype='hdf5')
#bootstrap_neg2logL, bootstrap_b_neg2logL = lee.get_bootstrap_sample('all',batch_size=100,dbtype='sqlite')
bootstrap_chi2 = bootstrap_b_neg2logL - bootstrap_neg2logL

df_null, df_null_obs = lee.load_results(lee.local_table+lee.nullname,['neg2logL'],get_observed=True)
df_prof, df_prof_obs = lee.load_results(lee.profiled_table,['neg2logL_quad','logw'],get_observed=True)
chi2_quad     = df_null['neg2logL'] - df_prof['neg2logL_quad']
chi2_quad_obs = df_null_obs['neg2logL'][0] - df_prof_obs['neg2logL_quad'][0]
w = np.exp(df_prof['logw'])
#chi2_quad = df['neg2logL_profiled_quad']

# Plots!

# Local histograms, combined and for each analysis
for aname,a in list(lee.LEEanalyses.items()) + [('combined',lee)]:
    print("aname, a:", aname, a)
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax2.set(yscale='log')

    df_null = a.load_results(a.local_table+a.nullname,['neg2logL'])
    df_sig = a.load_results(a.local_table+sig_name,['neg2logL'])
  
    qb  = df_null['neg2logL']
    #qb_quad = q_quad[:,i].numpy()
    qsb = df_sig['neg2logL']
    if np.sum(np.isfinite(qb)) < 2:
        print("qb mostly nan!")
    if np.sum(np.isfinite(qsb)) < 2:
        print("qsb mostly nan!")
    qb = qb[np.isfinite(qb)]
    qsb = qsb[np.isfinite(qsb)]
    
    sns.distplot(qsb-qb, bins=50, color='b',kde=False, ax=ax1, norm_hist=True)
    sns.distplot(qsb-qb, bins=50, color='b',kde=False, ax=ax2, norm_hist=True)
   
    # Compute and plot asymptotic distributions!
    df_A = a.load_results(a.local_table+sig_name,['qAsb','qAb','qO'],from_observed=True)
    qAsb = df_A['qAsb'][0]
    qAb  = df_A['qAb'][0]
    qO   = df_A['qO'][0]
    
    var_mu_sb = 1./tf.abs(qAsb) 
    var_mu_b  = 1./tf.abs(qAb) 
    
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
    apval = tfd.Normal(0,1).cdf(np.abs(qO - Eq_b) / np.sqrt(Vq_b))
    asig = -tfd.Normal(0,1).quantile(apval)
    
    sns.lineplot(qbx,qby,color='b',ax=ax1)
    sns.lineplot(qsbx,qsby,color='r',ax=ax1)
    
    sns.lineplot(qbx, qby,color='b',ax=ax2)
    sns.lineplot(qsbx,qsby,color='r',ax=ax2)
    
    #print("qO[{0}]: {1}".format(i,qO[i]))
    
    ax1.axvline(x=qO,lw=2,c='k',label="apval={0}, z={1:.1f}".format(apval,asig))
    ax2.axvline(x=qO,lw=2,c='k',label="apval={0}, z={1:.1f}".format(apval,asig))
    
    ax1.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
    ax2.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)

    fig.tight_layout()
    fig.savefig("{0}/local_hist_{1}_{2}.png".format(path,sig_name,aname))

# Global/profiled histograms
fig  = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax2.set(yscale="log")
sns.distplot(chi2_quad, color='m', kde=False, ax=ax1, norm_hist=True, label="LEEC quad", hist_kws={'weights': w})
sns.distplot(chi2_quad, color='m', kde=False, ax=ax2, norm_hist=True, label="LEEC quad", hist_kws={'weights': w})
sns.distplot(bootstrap_chi2, color='b', kde=False, ax=ax1, norm_hist=True, label="LEEC bootstrap")
sns.distplot(bootstrap_chi2, color='b', kde=False, ax=ax2, norm_hist=True, label="LEEC bootstrap")
   
qx = np.linspace(0, np.max(chi2_quad),1000) # 6 sigma too far for tf, cdf is 1. single-precision float I guess
qy = tf.math.exp(tfd.Chi2(df=DOF).log_prob(qx))
sns.lineplot(qx,qy,color='g',ax=ax1, label="asymptotic")
sns.lineplot(qx,qy,color='g',ax=ax2, label="asymptotic")

# Observed empirical and asymptotic p-values
epval = 1 - CDFf(chi2_quad)(chi2_quad_obs)
apval = tfd.Chi2(df=DOF).log_prob(chi2_quad_obs)
esig = -tfd.Normal(0,1).quantile(epval)
asig = -tfd.Normal(0,1).quantile(apval)

ax1.axvline(x=chi2_quad_obs,lw=2,c='k',label="e_z={0}, a_z={1}".format(asig,esig))
ax2.axvline(x=chi2_quad_obs,lw=2,c='k',label="e_z={0}, a_z={1}".format(asig,esig))

ax1.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
ax2.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
   
fig.tight_layout()
fig.savefig("{0}/LEEC_quad_{1}_{2}.png".format(path,master_name,nullname))
