"""Testing LEEcorrector objects"""

from analysis import collider_analyses_from_long_YAML, JMCJoint, deep_merge, LEEcorrection, LEECorrectorAnalysis, LEECorrectorMaster
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

nosignal = {a.name: {'s': tf.constant([[0. for sr in a.SR_names]],dtype=float)} for a in analyses_read.values()}
DOF = 2

path = 'TEST'
master_name = 'all'
nullname = 'background'
lee = LEECorrectorMaster(analyses_read,path,master_name,nosignal,nullname)
lee.ensure_equal_events()
lee.add_events(int(1e4))
lee.process_null()
lee.process_signals(signals,new_events_only=True)
df = lee.load_results(lee.combined_table,['neg2logL_null','neg2logL_profiled_quad'])
print('neg2logL_null:',df['neg2logL_null'])
print('neg2logL_profiled_quad:',df['neg2logL_profiled_quad'])
chi2_quad = df['neg2logL_null'] - df['neg2logL_profiled_quad']
#chi2_quad = df['neg2logL_profiled_quad']

# Plots!
fig  = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax2.set(yscale="log")
sns.distplot(chi2_quad, color='m', kde=False, ax=ax1, norm_hist=True, label="LEEC quad")
sns.distplot(chi2_quad, color='m', kde=False, ax=ax2, norm_hist=True, label="LEEC quad")
   
qx = np.linspace(0, np.max(chi2_quad),1000) # 6 sigma too far for tf, cdf is 1. single-precision float I guess
qy = tf.math.exp(tfd.Chi2(df=DOF).log_prob(qx))
sns.lineplot(qx,qy,color='g',ax=ax1, label="asymptotic")
sns.lineplot(qx,qy,color='g',ax=ax2, label="asymptotic")

ax1.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
ax2.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
   
fig.tight_layout()
fig.savefig("{0}/LEEC_quad_{1}_{2}.png".format(path,master_name,nullname))
