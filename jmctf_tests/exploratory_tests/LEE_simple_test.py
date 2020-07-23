"""Simple test setup for LEE correction driver object 
"""

import numpy as np
from jmctf import NormalAnalysis, BinnedAnalysis, JointDistribution

# Create component analysis objects
x_obs = 5.
sigma = 2.
norm = NormalAnalysis("Test normal", x_obs, sigma)
analyses = [norm]

# For this test, the "mu" parameter for NormalAnalysis will be the
# one we use to create our set of alternate hypothesis

N = 100 # Number of alternate hypotheses to generate (controls density of hypotheses)
null_mu = 0
mu = np.linspace(null_mu - 5*sigma, null_mu + 5*sigma, N)

class SigGen:
    """Object to supply signal hypotheses in chunks
       Replace with something that e.g. reads from HDF5 file
       in real cases."""
    def __init__(self,N,alt_hyp):
        self.count = N
        self.chunk_size = 10
        self.alt_hyp = alt_hyp
        self.j = 0

    def reset(self): 
        self.j = 0

    def next(self):
        j = self.j
        size = self.chunk_size
        chunk = {name: {par: dat[j:j+size] for par,dat in a.items()} for name,a in self.alt_hyp.items()}
        self.j += size
        print("chunk:", chunk)
        return chunk, list(range(j,j+size))

import tensorflow as tf
import jmctf.common as c
null_hyp = {"Test normal": {"mu": 0}}
alt_hyp = {"Test normal": {"mu": mu}} 
DOF = 1

from jmctf.LEE import LEECorrectorMaster

path = 'Test_normal'
master_name = 'test_1'
nullname = 'mu=0'
lee = LEECorrectorMaster(analyses,path,master_name,null_hyp,nullname)

# Make sure we are providing all the required analysis parameters
free, fixed, nuis = lee.get_parameter_structure()
print("free:", free)
print("fixed:", fixed)
print("nuis:", nuis)
print("null_hyp:", null_hyp)
print("alt_hyp:", alt_hyp)

# Generate pseudodata samples from null hypothesis
# lee.add_events(1e4)

# Fit null hypothesis nuisance parameters to recorded samples
lee.process_null()

# Fit all alternate hypothesis nuisance parameters to recorded
# samples, recording results only for the one which is the best 
# fit for each sample (not feasible, or necessary, to record them all)
lee.process_alternate(SigGen(N,alt_hyp),new_events_only=True)

# Load up results from the output databases (as pandas dataframes)
df_null, df_null_obs = lee.load_results(lee.null_table,['neg2logL'])
df_prof, df_prof_obs = lee.load_results(lee.profiled_table,['neg2logL_quad','logw'])

# Compute likelihood ratios
LLR = df_null['neg2logL'] - df_prof['neg2logL_quad']

# Plot TS distribution 
fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111)
plot_chi2(ax,LLR,DOF) 
ax.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
fig.tight_layout()
fig.savefig("LEE_simple_test_plot_LLR.svg")
