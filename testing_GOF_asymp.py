"""Testing asympotitc behaviour of semi-MC look-elsewhere correction"""

from analysis import collider_analyses_from_long_YAML, JMCJoint, deep_merge, LEEcorrection
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
#s_in = [1.,10.,20.,50.]
nosignal = {a.name: {'s': tf.constant([[0. for sr in a.SR_names]],dtype=float)} for a in analyses_read.values()}
#signal = {a.name: {'s': tf.constant([[s for sr in a.SR_names] for s in s_in], dtype=float)} for a in analyses_read.values()}
nullnuis = {a.name: {'nuisance': None} for a in analyses_read.values()} # Use to automatically set nuisance parameters to zero for sample generation

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
sgrid = ndim_grid(start,stop,30)
#sigs = np.linspace(-4*bsysi,+4*bsysi,50)
#print("sigs:", sigs)
#np.random.shuffle(sigs)
#sgrid = tf.expand_dims(tf.constant(sigs,dtype=float),axis=1)
signal = {"TEST": {'s': tf.constant(sgrid,dtype=float)}}
Ns = len(sgrid)

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
 
        #LEEcorrection(analyses,signal,nosignal,name=a.name+"_fitall",N=N)
        LEEcorrection(analyses,signal,nosignal,name=a.name+"_quadonly",N=N,fitall=False)
