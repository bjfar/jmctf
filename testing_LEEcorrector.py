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
f = 'old_junk/CBit_analyses.yaml'
stream = open(f, 'r')
analyses_read, SR_map, iSR_map = collider_analyses_from_long_YAML(stream,replace_SR_names=True)

nosignal = {a.name: {'s': tf.constant([[0. for sr in a.SR_names]],dtype=float)} for a in analyses_read.values()}

for a in analyses_read.values():
    this_nosig = {a.name: nosignal[a.name]}
    lee = LEECorrectorAnalysis(a,'test_path','combined',this_nosig)
    lee.add_events(this_nosig,int(2e4))
    lee.process_background()
