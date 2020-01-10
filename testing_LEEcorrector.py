"""Testing LEEcorrector objects"""

from analysis import collider_analyses_from_long_YAML, JMCJoint, deep_merge, LEEcorrection, LEECorrectorAnalysis
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

Ns, signal = get_grid(analyses_read,10) # For testing only! Will die if used for more than e.g. 3 total SRs.

nosignal = {a.name: {'s': tf.constant([[0. for sr in a.SR_names]],dtype=float)} for a in analyses_read.values()}

for a in analyses_read.values():
    this_nosig = {a.name: nosignal[a.name]}
    lee = LEECorrectorAnalysis(a,'TEST','combined',this_nosig)
    lee.add_events(this_nosig,int(1e4))
    lee.process_background()
