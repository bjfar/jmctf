from analysis import collider_analyses_from_long_YAML, JMCJoint, deep_merge, LEEcorrection, LEECorrectorAnalysis, LEECorrectorMaster
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import h5py

analysis = 
# Load data from sql
path_sql = "/home/farmer/repos/JMCtf/TEST_EWMSSM_OLD"

# Load data from hdf5
path_hdf5 = "/home/farmer/repos/JMCtf/TEST_EWMSSM_OLD_3"
 

# Take the minimum over the null and profiled neg2logLs, to ensure that the null is treated
# as nested within the alternate. Needed when sampling of alternate isn't good enough that
# null is 'naturally' nested.
min_neg2logL = df[['neg2logL_null','neg2logL_profiled_quad']].min(axis=1)
chi2_quad = df['neg2logL_null'] - min_neg2logL

# Plots!
fig  = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax2.set(yscale="log")
sns.distplot(chi2_quad, color='m', kde=False, ax=ax1, norm_hist=True, label="LEEC quad")
sns.distplot(chi2_quad, color='m', kde=False, ax=ax2, norm_hist=True, label="LEEC quad")

qx = np.linspace(0, np.max(chi2_quad),1000) # 6 sigma too far for tf, cdf is 1. single-precision float I guess
qy = 0.5*tf.math.exp(tfd.Chi2(df=DOF).log_prob(qx)) # Half-chi2 since negative signal contributions not possible?
sns.lineplot(qx,qy,color='g',ax=ax1, label="asymptotic")
sns.lineplot(qx,qy,color='g',ax=ax2, label="asymptotic")

ax1.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
ax2.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
   
fig.tight_layout()
fig.savefig("{0}/LEEC_quad_debug_{1}_{2}.png".format(path,master_name,nullname))
