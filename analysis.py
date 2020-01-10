import numpy as np
import scipy.interpolate as spi
import scipy.stats as sps
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import pandas as pd
import sqlite3
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import massminimize as mm
import copy
import pathlib
import time

# Stuff to help format YAML output
class blockseqtrue( list ): pass
def blockseqtrue_rep(dumper, data):
        return dumper.represent_sequence( u'tag:yaml.org,2002:seq', data, flow_style=True )
yaml.add_representer(blockseqtrue, blockseqtrue_rep)

def eCDF(x):
    """Get empirical CDFs of arrays of samples. Assumes first dimension
       is the sample dimension. All CDFs are the same since number of
       samples has to be the same"""
    cdf = tf.constant(np.arange(1, x.shape[0]+1)/float(x.shape[0]),dtype=float)
    #print("cdf.shape:", cdf.shape)
    #print("x.shape:", x.shape)
    return cdf
    #return tf.broadcast_to(cdf,x.shape)

def CDFf(samples,reverse=False):
    """Return interpolating function for CDF of some simulated samples"""
    if reverse:
        s = np.argsort(samples[np.isfinite(samples)],axis=0)[::-1] 
    else:
        s = np.argsort(samples[np.isfinite(samples)],axis=0)
    ecdf = eCDF(samples[s])
    CDF = spi.interp1d([-1e99]+list(samples[s])+[1e99],[ecdf[0]]+list(ecdf)+[ecdf[1]])
    return CDF, s #pvalue may be 1 - CDF(obs), depending on definition/ordering

def gather_by_idx(x,indices):
    idx = tf.cast(indices,dtype=tf.int32)
    idx_flattened = tf.range(0, x.shape[0]) * x.shape[1] + idx
    y = tf.gather(tf.reshape(x, [-1]),  # flatten input
                  idx_flattened)  # use flattened indices
    return y

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

# Want to convert all this to YAML. Write a simple container to help with this.
class ColliderAnalysis:
    def __init__(self,name,srs=None,cov=None,cov_order=None,unlisted_corr_zero=False,verify=True):
        self.name = name
        self.cov = cov
        self.cov_order = cov_order
        self.unlisted_corr_zero = unlisted_corr_zero
        if srs is None:
            self.SR_names = None
            self.SR_n = None
            self.SR_b = None
            self.SR_b_sys = None
        else:
            self.SR_names = [None]*len(srs)
            self.SR_n     = [None]*len(srs)
            self.SR_b     = [None]*len(srs)
            self.SR_b_sys = [None]*len(srs)
            for i,sr in enumerate(srs):
                self.SR_names[i] = sr[0]
                self.SR_n[i]     = sr[1]
                self.SR_b[i]     = sr[2]
                self.SR_b_sys[i] = sr[3]
        self.SR_n = np.array(self.SR_n)
        self.SR_b = np.array(self.SR_b)
        self.SR_b_sys = np.array(self.SR_b_sys)
 
        # Scaling factors to scale canonical input parameters to parameters with ~variance=1 MLEs
        self.s_scaling = np.sqrt(self.SR_b + self.SR_b_sys**2)
        self.theta_scaling = self.SR_b_sys #/ 10. # Factor 10 experimental 

        # Covariance matrix selection and ordering
        self.cov_order = self.get_cov_order()
        if self.cov is not None:
            self.in_cov = np.array([1 if sr in self.cov_order else 0 for sr in self.SR_names], dtype=np.bool)
            self.covi = [self.SR_names.index(sr) for sr in self.cov_order]
            self.cov_diag = [self.cov[k][k] for k in range(len(self.cov))]
 
        if verify: self.verify() # Set this flag zero for "manual" data input
        # Mega-simple bin-by-bin significance estimate, for cross-checking
        # print("Analysis {0}: significance per SR:".format(self.name))
        # for i,sr in enumerate(self.SR_names):
        #     print("   {0}: {1:.1f}".format(sr, np.abs(self.SR_n[i] - self.SR_b[i])/np.sqrt(self.SR_b[i] + self.SR_b_sys[i]**2)))

    def get_cov_order(self):
        cov_order = None
        if self.cov is not None:
            if self.cov_order is None:
                raise ValueError("Covariance matrix supplied, but signal region ordering has not been specified!")
            if self.cov_order=="use SR order": 
                cov_order = self.SR_names 
            else:
                cov_order = self.cov_order
        return cov_order

    def verify(self):
        """Check that internal data makes sense, and warn user about potential issues"""
        #for i, sr in enumerate(self.SR_names):
        #    if self.SR_b[i]==0:
        #        print("WARNING! Expected background is exactly zero for region {0} in analysis {1}. This doesn't make sense. We will add a tiny offset to make it numerically tractable, but please check your input data!".format(sr, self.name)) 
        #        self.SR_b[i] += 1e-10
        # Actually no, should be ok given the systematic.

    def tensorflow_model(self,pars):
        """Output tensorflow probability model object, to be combined together and
           sampled from.
           pars       - dictionary of signal and nuisance parameters (tensors, constant or Variable)
        """

        # Need to construct these shapes to match the event_shape, batch_shape, sample_shape 
        # semantics of tensorflow_probability.

        cov_order = self.get_cov_order()
        small = 1e-10
        #print("pars:",pars)
        tfds = {}

        # Determine which SRs participate in the covariance matrix
        if self.cov is not None:
            cov = tf.constant(self.cov,dtype=float)
            cov_diag = tf.constant([self.cov[k][k] for k in range(len(self.cov))])
            # Select which systematic to use, depending on whether SR participates in the covariance matrix
            bsys_tmp = [np.sqrt(self.cov_diag[cov_order.index(sr)]) if self.in_cov[i] else self.SR_b_sys[i] for i,sr in enumerate(self.SR_names)]
        else:
            bsys_tmp = self.SR_b_sys[:]

        # Prepare input parameters
        #print("input pars:",pars)
        b = tf.expand_dims(tf.constant(self.SR_b,dtype=float),0) # Expand to match shape of signal list
        bsys = tf.expand_dims(tf.constant(bsys_tmp,dtype=float),0)
        s = pars['s'] * self.s_scaling # We "scan" normalised versions of s, to help optimizer
        theta = pars['theta'] * self.theta_scaling # We "scan" normalised versions of theta, to help optimizer
        #print("de-scaled pars: s    :",s)
        #print("de-scaled pars: theta:",theta)
        theta_safe = theta
  
        #print("theta_safe:", theta_safe)
        #print("rate:", s+b+theta_safe)
 
        # Poisson model
        poises0  = tfd.Poisson(rate = tf.abs(s+b+theta_safe)+1e-10) # Abs works to constrain rate to be positive. Might be confusing to interpret BF parameters though.
        # Treat SR batch dims as event dims
        poises0i = tfd.Independent(distribution=poises0, reinterpreted_batch_ndims=1)
        tfds["{0}::n".format(self.name)] = poises0i

        # Multivariate background constraints
        if self.cov is not None:
            #print("theta_safe:",theta_safe)
            #print("covi:",self.covi)
            theta_cov = tf.gather(theta_safe,self.covi,axis=-1)
            #print("theta_cov:",theta_cov)
            cov_nuis = tfd.MultivariateNormalFullCovariance(loc=theta_cov,covariance_matrix=cov)
            tfds["{0}::x_cov".format(self.name)] = cov_nuis
            #print("str(cov_nuis):", str(cov_nuis))

            # Remaining uncorrelated background constraints
            if np.sum(~self.in_cov)>0:
                nuis0 = tfd.Normal(loc = theta_safe[...,~self.in_cov], scale = bsys[...,~self.in_cov])
                # Treat SR batch dims as event dims
                nuis0i = tfd.Independent(distribution=nuis0, reinterpreted_batch_ndims=1)
                tfds["{0}::x_nocov".format(self.name)] = nuis0i
        else:
            # Only have uncorrelated background constraints
            nuis0 = tfd.Normal(loc = theta_safe, scale = bsys)
            # Treat SR batch dims as event dims
            nuis0i = tfd.Independent(distribution=nuis0, reinterpreted_batch_ndims=1)
            tfds["{0}::x".format(self.name)] = nuis0i 
        #print("hello3")

        return tfds #, sample_layout, sample_count

    def get_Asimov_samples(self,signal_pars):
        """Construct 'Asimov' samples for this analysis
           Used to detemine asymptotic distribtuions of 
           certain test statistics.

           Assumes target MLE value for nuisance 
           parameters is zero.
        """
        Asamples = {}
        s = signal_pars['s'] * self.s_scaling
        b = tf.expand_dims(tf.constant(self.SR_b,dtype=float),0) # Expand to match shape of signal list 
        #print("Asimov s:",s)
        #print("self.in_cov:", self.in_cov)
        Asamples["{0}::n".format(self.name)] = tf.expand_dims(b + s,0) # Expand to sample dimension size 1
        if self.cov is not None:
            Asamples["{0}::x_cov".format(self.name)] = tf.constant(np.zeros((1,s.shape[0],np.sum(self.in_cov))),dtype=float)
            if np.sum(~self.in_cov)>0:
                Asamples["{0}::x_nocov".format(self.name)] = tf.constant(np.zeros((1,s.shape[0],np.sum(~self.in_cov))),dtype=float)
        else:
            Asamples["{0}::x".format(self.name)] = tf.expand_dims(0*s,0)
        #print("{0}: Asamples: {1}".format(self.name, Asamples))
        return Asamples

    def get_observed_samples(self):
        """Construct dictionary of observed data for this analysis"""
        Osamples = {}
        Osamples["{0}::n".format(self.name)] = tf.expand_dims(tf.expand_dims(tf.constant(self.SR_n,dtype=float),0),0)
        if self.cov is not None:
            Osamples["{0}::x_cov".format(self.name)] = tf.expand_dims(tf.expand_dims(tf.constant([0]*np.sum(self.in_cov),dtype=float),0),0)
            if np.sum(~self.in_cov)>0:
                Osamples["{0}::x_nocov".format(self.name)] = tf.expand_dims(tf.expand_dims(tf.constant([0]*np.sum(~self.in_cov),dtype=float),0),0)
        else:
            Osamples["{0}::x".format(self.name)] = tf.expand_dims(tf.expand_dims(tf.constant([0]*len(self.SR_names),dtype=float),0),0)
        return Osamples

    def get_sample_structure(self):
        """Get a dictionary describing the structure of data samples for this analysis.
           Basically just the keys of the sample dictionaries plus dimension of each entry"""
        data = self.get_observed_samples() # Might as well infer it from this data
        structure = {key.split("::")[1]: val.shape[-1] for key,val in data.items()} # self.name part stripped out of key for brevity 
        return structure

    def get_nuisance_parameter_structure(self):
        """Get a dictionary describing the nuisance parameter structure of this analysis.
           Basically just the keys of the parameter dictionaries plus dimension of each entry"""
        return {"theta": len(self.SR_b)} # Just one nuisance parameter per signal region, packaged into one tensor. 

    def get_nuisance_tensorflow_variables(self,sample_dict,signal):
        """Get nuisance parameters to be optimized, for input to "tensorflow_model"""
        seeds = self.get_seeds_nuis(sample_dict,signal) # Get initial guesses for nuisance parameter MLEs
        stacked_seeds = np.stack([seeds[sr]['theta'] for sr in self.SR_names],axis=-1)
        thetas = {"theta": tf.Variable(stacked_seeds, dtype=float, name='theta')}
        return thetas

    def get_all_tensorflow_variables(self,sample_dict):
        """Get all parameters (signal and nuisance) to be optimized, for input to "tensorflow_model"""
        seeds = self.get_seeds_s_and_nuis(sample_dict) # Get initial guesses for parameter MLEs
        stacked_theta = np.stack([seeds[sr]['theta'] for sr in self.SR_names],axis=-1)
        stacked_s     = np.stack([seeds[sr]['s'] for sr in self.SR_names],axis=-1)
        pars = {"s": tf.Variable(stacked_s, dtype=float, name='s'),
                "theta": tf.Variable(stacked_theta, dtype=float, name='theta')}
        return pars
        
    def as_dict_short_form(self):
        """Add contents to dictionary, ready for dumping to YAML file
           Compact format version"""
        tmpd = {}
        tmpd["Type"] = "Poisson_with_Multinormal_Nuisance"
        tmpd["SR_names"] = self.SR_names # Maybe leave this as "-" items rather than single-line format
        tmpd["counts"] = blockseqtrue(a.SR_n)
        tmpd["background"] = blockseqtrue(a.SR_b)
        tmpd["background_sys_uncert"] = blockseqtrue(a.SR_b_sys)
        if self.cov is not None:
            tmpd["cov"] = [blockseqtrue(row) for row in self.cov]
        else:
            tmpd["cov"] = None
        return tmpd

    def as_dict_long_form(self):
        """Add contents to dictionary, ready for dumping to YAML file
           Long format version"""
        tmpd = {}
        tmpd["Type"] = "Poisson_with_Multinormal_Nuisance"
        srs = []
        for name, n, b, bsys in zip(self.SR_names,self.SR_n,self.SR_b,self.SR_b_sys):
            srs += [blockseqtrue([name, n, b, bsys])]
        tmpd["Signal regions"] = srs
        if self.cov is not None:
            tmpd["cov"] = [blockseqtrue(row) for row in self.cov]
            if self.cov_order is None:
                tmpd["cov_order"] = "use SR order"
            else:
                tmpd["cov_order"] = self.cov_order
        tmpd["unlisted_corr_zero"] = self.unlisted_corr_zero
        return tmpd

    def as_dataframe(self):
        """Extract contents into a Pandas dataframe for nice viewing
           Not including covariance matrix for now.
        """
        d = self.as_dict_long_form()
        cols=["SR","n","b","b_sys"]
        df = pd.DataFrame(columns=cols)
        df.set_index('SR',inplace=True)
        for data in d["Signal regions"]:
            df.loc[data[0]] = data[1:]
        return df

    def get_seeds_s_and_nuis(self,samples):
        """Get seeds for full fit to free signal and nuisance
           parameters for every SR. Gives exact MLEs in the
           absence of correlations"""
        seeds={}

        threshold = 1e-4 # Smallness threshold, for fixing numerical errors and disallowing solutions too close to zero

        n_all = samples['{0}::n'.format(self.name)]
        if self.cov is not None:
            xcov_all = samples['{0}::x_cov'.format(self.name)]
            if np.sum(~self.in_cov)>0:
                xnocov_all = samples['{0}::x_nocov'.format(self.name)]
        else:
            x_all = samples['{0}::x'.format(self.name)]

        for i,sr in enumerate(self.SR_names): 
            #print("Getting seeds for analysis {0}, region {1}".format(self.name,sr))
            seeds[sr] = {}
            # From input
            n = n_all[...,i] 
            if self.cov is not None:
                if sr in self.cov_order:
                    x = xcov_all[...,i]
                else:
                    x = xnocov_all[...,i]
            else:
                x = x_all[...,i]
       
            # From object member variables
            b = tf.constant(self.SR_b[i],dtype=float)
            bsys = tf.constant(self.SR_b_sys[i],dtype=float)

            bcast = np.ones((x+b+n).shape) # For easier broadcasting
 
            theta_MLE_tmp = x*bcast
            s_MLE = (n - x - b)*bcast
            l_MLE_tmp = (s_MLE + b + theta_MLE_tmp)*bcast
            # Sometimes get l < 0 predictions just due to small numerical errors. Fix these if they
            # are small enough not to matter
            mfix = (l_MLE_tmp.numpy()<threshold) & (l_MLE_tmp.numpy()>-threshold) # fix up these ones with small adjustment

            #print("theta_MLE:",theta_MLE_tmp)
            #print("s_MLE:",s_MLE)
            #print("n:", n)
            #print("b:",b)
            #print("l_MLE:",l_MLE_tmp)
            #print("mfix:",mfix)

            theta_MLE = np.zeros(theta_MLE_tmp.shape)
            theta_MLE[mfix] = x.numpy()[mfix] + 2*threshold
            theta_MLE[~mfix] = x.numpy()[~mfix]

            acheck = (s_MLE + b + theta_MLE).numpy()<threshold # Should have been fixed
            # Error if too negative. rate MLE should never be negative.
            if np.sum(acheck)>0:
                print("Negative rate MLEs!:")
                print("theta_MLE:",theta_MLE[acheck])
                print("s_MLE:",s_MLE[acheck])
                print("l_MLE:",(s_MLE+b+theta_MLE)[acheck])
                raise ValueError("{0} negative rate MLEs detected!".format(np.sum(acheck)))

            seeds[sr]['theta'] = theta_MLE / self.theta_scaling[i]
            seeds[sr]['s']     = s_MLE / self.s_scaling[i]

            #print("seeds[{0}], theta: {1}".format(sr,seeds[sr]['theta']))
            #print("seeds[{0}], s    : {1}".format(sr,seeds[sr]['s']))

        return seeds


    def get_seeds_nuis(self,samples,signal_pars):
        """Get seeds for (additive) nuisance parameter fits,
           assuming fixed signal parameters. Gives exact MLEs in
           the absence of correlations
           TODO: Check that signal parameters are, in fact, fixed?
           """
        verbose = False # For debugging
        if verbose: print("signal (seeds_nuis):",signal)
        seeds={}
        self.theta_both={} # For debugging, need to compare BOTH solutions to numerical MLEs.
        self.theta_dat={}

        threshold = 1e-4 # Smallness threshold, for fixing numerical errors and disallowing solutions too close to zero

        n_all = samples['{0}::n'.format(self.name)]
        if self.cov is not None:
            xcov_all = samples['{0}::x_cov'.format(self.name)]
            if np.sum(~self.in_cov)>0:
                xnocov_all = samples['{0}::x_nocov'.format(self.name)]
        else:
            x_all = samples['{0}::x'.format(self.name)]
        s_all = signal_pars['s'] # non-scaled! 
 
        #print("s_all:", s_all)
        #print("n_all:", n_all)
        #print("x_all:", x_all)

        # TODO: Could perhaps do all SRs at once, but that's a bit of a pain and this is fast enough anyway
        for i,sr in enumerate(self.SR_names): 
            #print("Getting seeds for analysis {0}, region {1}".format(self.name,sr))
            seeds[sr] = {}
            # From input
            n = n_all[...,i] 
            if self.cov is not None:
                if sr in self.cov_order:
                    x = xcov_all[...,i]
                else:
                    x = xnocov_all[...,i]
            else:
                x = x_all[...,i]
            s = s_all[...,i]

            #print("s:", s)
            #print("n:", n)
            #print("x:", x)

            # From object member variables
            b = tf.constant(self.SR_b[i],dtype=float)
            bsys = tf.constant(self.SR_b_sys[i],dtype=float)

            #print("s:", s)
            #print("b:", b)
            #print("x:", x)            

            # Math!
            A = 1./bsys**2
            B = 1 + A*(s + b - x)
            C = (s+b)*(1-A*x) - n
            D = (B**2 - 4*A*C).numpy() # to be used in masks
            #D<0:
            #print("D:", D.shape)
            #print("D<0", (D<0).shape)
            bcast = np.ones((s+b+n).shape) # For easier broadcasting
            theta_MLE = np.zeros(bcast.shape)
            #print("theta_MLE", theta_MLE.shape)
            #print("theta_MLE[D<0]", theta_MLE[D<0])
            #print("s+b", (s+b).shape)
            if verbose: print("broadcast:", (-(s+b)*bcast).shape)
            #print("broadcast[D<0]:", (-(s+b)*bcast)[D<0].shape)
            # No solutions! This is a drag, means MLE is on boundary of allowed space, not at a 'real' minima.
            # Will mess up Wilk's theorem a bit. However MLE will necessarily just be on the boundary
            # s + b + theta = 0
            # so it is easy to compute at least
            if verbose: print("s:",s)
            if verbose: print("b:",b)
            if verbose: print("D:", D.shape)
            theta_MLE[D<0] = (-(s+b)*bcast)[D<0]
            if verbose: print("No solution count:", np.sum(D<0))
            #elif D==0:
            # One real solution:
            #print("B:",B)
            #print("A:",A)
            #print("D:",D)
            #print("bcast.shape:", bcast.shape)
            theta_MLE[D==0] = -B[D==0]/(2*A) # Is this always positive? Not sure...
            if verbose: print("Single solution count:", np.sum(D==0))
            #elif D>0:
            # Two real solutions
            # Hmm not sure how to pick... I guess first check that they are indeed in the "allowed" region
            if verbose: print("Two solution count:", np.sum(D>0))
            r1 = (-B + np.sqrt(D))/(2*A)
            r2 = (-B - np.sqrt(D))/(2*A)
            a1 = (s+b+r1 >= 0)
            a2 = (s+b+r2 >= 0) 
            # See if just one is allowed
            ma= ((D>0) & a1 & a2).numpy()
            mf= ((D>0) & (~a1) & (~a2)).numpy()
            # m = ma | mf 
            if verbose: print("   both allowed:", np.sum(ma))
            if verbose: print("   both forbidden:", np.sum(mf))
            # #print(list(zip(n[D>0]-s-b,r1,r2)))
            # #if both allowed, pick the one closest to the MLE for theta that one would get without the nuisance constraint
            # Actually just throw error, I think this should not happen
            #if np.sum(ma)>0:
            #    for r1i, r2i in zip(r1[ma],r2[ma]):
            #        print("r1: {0}, r2: {1}, s+b: {2}".format(r1i,r2i,s+b))
            #    raise ValueError("Found multiple allowed solutions for some MLEs! I'm pretty sure this shouldn't happen so I'm calling it an error/bug in the calculation") 
            # If both are forbidden, make sure that it isn't just by some tiny amount due to numerical error.
            fix1 = (~a1 & ((s+b)*bcast+r1 >= -np.abs(r1)*threshold)).numpy() 
            fix2 = (~a2 & ((s+b)*bcast+r2 >= -np.abs(r2)*threshold)).numpy()
            #m12 = (D>0) & fix1 & fix2 # Don't need to worry about this, means they are probably the same solution anyway
            m1  = (D>0) & mf & fix1
            m2  = (D>0) & mf & fix2
            #print("Corrected {0} positive solutions".format(np.sum(m1)))
            #print("m1",m1.shape)
            #print("r1",r1.shape)
            #print("(s+b)*bcast",((s+b)*bcast).shape)
            #print("theta_MLE", theta_MLE.shape)
            #print("theta_MLE[m1]",theta_MLE[m1])
            theta_MLE[m1] = -((s+b)*bcast + np.abs(r1)*threshold)[m1] # Make sure still positive after numerics
            theta_MLE[m2] = -((s+b)*bcast + np.abs(r2)*threshold)[m2]
            #if np.sum(mf & ~(fix1 | fix2) > 0):
            #    for r1i, r2i in zip(r1[mf],r2[mf]):
            #        print("r1: {0}, r2: {1}, s+b: {2}".format(r1i,r2i,s+b))
            #    raise ValueError("Found both solutions forbidden (and unfixable) for some MLEs! I'm pretty sure this shouldn't happen so I'm calling it an error/bug in the calculation") 
            # Edit: I think the above can happen for super low background signal regions. Then the nuisance fluctuation can make even the effective background estimate negative. So I think here we just have to stick the MLE on the boundary of the parameter space.
            mbound = mf & ~(fix1 | fix2)
            theta_MLE[mbound] = (-(s+b)*bcast + threshold)[mbound]

            #d1 = (r1 - (n - s - b))**2
            #d2 = (r2 - (n - s - b))**2
            # Or the closest to the MLE using a normal approximation for the Poisson? And fixing variance with theta=0...
            mzero = ((s+b)*bcast == 0).numpy()
            #print("mzero:",mzero)
            MLE_norm = ((x*A + (n-s-b)/(s+b)) / (A + 1./(s+b))).numpy()
            MLE_norm[mzero] = 0 
            d1 = (r1 - MLE_norm)**2
            d2 = (r2 - MLE_norm)**2
            # Use this if both solutions are allowed.
            m1 = ((D>0) & a1 & a2 & (d1<d2)).numpy()
            m2 = ((D>0) & a1 & a2 & (d1>=d2)).numpy()
            if verbose: print("   both roots allowed, but positive is closer to gaussian estimate:", np.sum(m1))
            if verbose: print("   both roots allowed, but negative is closer to gaussian estimate:", np.sum(m2))
            theta_MLE[m1] = r1[m1]
            theta_MLE[m2] = r1[m2]
            # or just smallest in magnitude?
            #d1 = r1**2
            #d2 = r2**2
            # The most proper thing is just to see which solution gives the best likelihood (or minimum log-likelihood)
            #d1 = -sps.poisson.logpmf(n,s+b+r1) - sps.norm.logpdf(x,r1)
            #d2 = -sps.poisson.logpmf(n,s+b+r2) - sps.norm.logpdf(x,r2)
            #d1[~np.isfinite(d1)] = 1e99 # bad likelihood
            #d2[~np.isfinite(d2)] = 1e99
            #theta_MLE[(D>0) & (d1<d2)] = r1[(D>0) & (d1<d2)]
            #theta_MLE[(D>0) & (d1>=d2)] = r2[(D>0) & (d1>=d2)]
            #print("r1", "r2", "MLE_guess")
            #for j in range(np.sum(D>0)):
            #    print(r1[j], r2[j], MLE_norm[j])
            #theta_MLE[D>0] = MLE_norm[D>0] # test...
            # Seems like the positive root is always the right one for some reason...
            #theta_MLE[D>0] = r1[D>0] 
            # # If both solutions are forbidden then something weird is probably going on, but we'll pick one anyway
            # # and let the parameter mapping sort it out.
            m = ((D>0) & (a1) & (~a2)).numpy()
            if verbose: print("   only positive root allowed:", np.sum(m))
            theta_MLE[m] = r1[m]
            m = ((D>0) & (~a1) & (a2)).numpy()
            if verbose: print("   only negative root allowed:", np.sum(m))
            theta_MLE[m] = r2[m]
            # Save some extra info for diagnositic functions
            self.theta_both[sr] = [copy.copy(theta_MLE), copy.copy(theta_MLE), MLE_norm, s + b] # Make sure to copy...
            self.theta_both[sr][0][D>0] = r1[D>0]
            self.theta_both[sr][1][D>0] = r2[D>0]
            self.theta_dat[sr] = n,x,s,b,bsys
            # Output!
            if verbose: print("MLE theta_{0}:".format(i), theta_MLE) 
            # Sanity check; are all these seeds part of the allowed parameter space?
            l = s + b + theta_MLE
            afix = ((l<=threshold) & (l>=-threshold)).numpy() # really small values can cause problems too.
            theta_MLE[afix] = theta_MLE[afix] + threshold
            acheck = (l<-threshold).numpy() # Raise error if too far negative to correct without problems
            #print("sr:", sr)
            #print("l:", l)
            # TODO: Put this check back in! Might be some error in here because it keeps triggering. But might be due to covariance matrices?
            #if np.sum(acheck)>0:
            #     for soli, r1i, r2i in zip(theta_MLE[acheck],r1[acheck],r2[acheck]):
            #        print("MLE: {0}, r1: {1}, r2: {2}, s+b: {3}, ? {4}".format(soli,r1i,r2i,s+b,s+b+r1i >= -np.abs(r1i)*1e-6))
            #     raise ValueError("Computed {0} forbidden seeds (from {1} samples)! There is therefore a bug in the seed calculations".format(np.sum(acheck),acheck.shape))
            # Did we get everything? Are there any NaNs left?
            nnans = np.sum(~np.isfinite(theta_MLE))
            if nnans>0: print("Warning! {0} NaNs left in seeds!".format(nnans))
            seeds[sr]['theta'] = theta_MLE / self.theta_scaling[i] # Scaled by bsys to try and normalise variables in the fit. Input variables are scaled the same way.
        #print("seeds:", seeds)
        #quit()
        return seeds

def neg2LogL(pars,const_pars,analyses,data,pre_scaled_pars,transform=None):
    """General -2logL function to optimise"""
    if transform is not None:
        pars_t = transform(pars)
    else:
        pars_t = pars
    if const_pars is None:
        all_pars = pars_t
    else:
        all_pars = deep_merge(const_pars,pars_t)
    #print("all_pars:",all_pars)
    joint = JMCJoint(analyses,all_pars,pre_scaled_pars)
    q = -2*joint.log_prob(data)
    total_loss = tf.math.reduce_sum(q)
    return total_loss, q, None, None

def optimize(pars,const_pars,analyses,data,pre_scaled_pars,transform=None,log_tag=''):
    """Wrapper for optimizer step that skips it if the initial guesses are known
       to be exact MLEs"""

    opts = {"optimizer": "Adam",
            "step": 0.05,
            "tol": 0.01,
            "grad_tol": 1e-4,
            "max_it": 100,
            "max_same": 5,
            'log_tag': log_tag 
            }

    kwargs = {'const_pars': const_pars,
              'analyses': analyses,
              'data': data,
              'pre_scaled_pars': pre_scaled_pars,
              'transform': transform
              }

    exact_MLEs = False #True
    for a in analyses.values():
        if a.cov is not None: exact_MLEs = False # TODO: generalise exactness determination
    if exact_MLEs:
        print("All starting MLE guesses are exact: skipping optimisation") 
        total_loss, q, none, none = neg2LogL(pars,**kwargs)
    else:
        print("Beginning optimisation")
        #f = tf.function(mm.tools.func_partial(neg2LogL,**kwargs))
        f = mm.tools.func_partial(neg2LogL,**kwargs)
        q, none, none = mm.optimize(pars, f, **opts)
    # Rebuild distribution object with fitted parameters
    if transform is not None:
        pars_t = transform(pars)
    else:
        pars_t = pars
    if const_pars is None:
        all_pars = pars_t
    else:
        all_pars = deep_merge(const_pars,pars_t)
    joint = JMCJoint(analyses,all_pars,pre_scaled_pars)
    return joint, q

class JMCJoint(tfd.JointDistributionNamed):
    """Object to combine analyses together and treat them as a single
       joint distribution. Uses JointDistributionNamed for most of the
       underlying work.
       
       analyses - dict of analysis-like objects to be combined
       signal - dictionary of dictionaries containing signal parameter
        or constant tensors for each analysis.
       model_type - Specify treame
       pre_scaled_pars - If true, all input parameters are already scaled such that MLEs
                         have variance of approx. 1 (for more stable fitting).
                         If false, parameters are conventionally (i.e. not) scaled, and
                         required scaling internally.
                       - 'all', 'nuis', None
    """
   
    def __init__(self, analyses, pars=None, pre_scaled_pars=None):
        self.analyses = analyses
        if pars is not None:
            self.pars = self.prepare_pars(pars,pre_scaled_pars)
            dists = {} 
            self.Asamples = {}
            self.Osamples = {}
            for a in analyses.values():
                d = a.tensorflow_model(self.pars[a.name])
                dists.update(d)
                self.Asamples.update(a.get_Asimov_samples(self.pars[a.name]))
                self.Osamples.update(a.get_observed_samples())
            super().__init__(dists) # Doesn't like it if I use self.dists, maybe some construction order issue...
            self.dists = dists
        # If no pars provided can still fit the analyses, but obvious cannot sample or compute log_prob etc.
        # TODO: can we fail more gracefully if people try to do this?
        #       Or possibly the fitting stuff should be in a different object? It seems kind of nice here though.

    def prepare_pars(self,pars,pre_scaled_pars):
        """Prepare default nuisance parameters and return scaled signal and nuisance parameters for each analysis
           (scaled such that MLE's in this parameterisation have
           variance of approx. 1"""
        scaled_pars = {}
        #print("pars:",pars)
        for aname,a in self.analyses.items():
            pardict = pars[aname]
            scaled_pars[aname] = {}
            if 'nuisance' in pardict.keys() and pardict['nuisance'] is None:
                # trigger shortcut to set nuisance parameters to zero, for sample generation. 
                theta_in = tf.constant(0*pardict['s'])
            else:
                theta_in = pardict['theta']
            if pre_scaled_pars is None:
                #print("Scaling input parameters...")
                scaled_pars[aname]['s']     = pardict['s'] / a.s_scaling
                scaled_pars[aname]['theta'] = theta_in / a.theta_scaling
            elif pre_scaled_pars=='nuis':
                #print("Scaling only signal parameters: nuisanced parameters already scaled...")
                scaled_pars[aname]['s']     = pardict['s'] / a.s_scaling
                scaled_pars[aname]['theta'] = theta_in
            elif pre_scaled_pars=='all':
                #print("No scaling applied: all parameters already scaled...")
                scaled_pars[aname]['s']     = pardict['s']
                scaled_pars[aname]['theta'] = theta_in
            else:
                raise ValueError("Invalid value of 'pre_scaled_pars' option! Please choose one of (None,'all','nuis)")
        return scaled_pars 

    def descale_pars(self,pars):
        """Remove scaling from parameters. Assumes they have all been scaled and require de-scaling."""
        descaled_pars = {}
        for ka,a in self.analyses.items():
          if ka in pars.keys():
            pardict = pars[ka]
            descaled_pars[ka] = {}
            if 's' in pardict.keys():     descaled_pars[ka]['s']     = pardict['s'] * a.s_scaling
            if 'theta' in pardict.keys(): descaled_pars[ka]['theta'] = pardict['theta'] * a.theta_scaling
        return descaled_pars

    def get_nuis_parameters(self,signal,samples):
        """Samples vector and signal provided to compute good starting guesses for parameters"""
        pars = {}
        for a in self.analyses.values():
            pars[a.name] = a.get_nuisance_tensorflow_variables(samples,signal[a.name])
        return pars

    def get_all_parameters(self,samples):
        """Samples vector and signal provided to compute good starting guesses for parameters"""
        pars = {}
        for a in self.analyses.values():
            pars[a.name] = a.get_all_tensorflow_variables(samples)
        return pars

    def fit_nuisance(self,signal,samples,log_tag=''):
        """Fit nuisance parameters to samples for a fixed signal
           (ignores parameters that were used to construct this object)"""
        nuis_pars = self.get_nuis_parameters(signal,samples)
        joint_fitted, q = optimize(nuis_pars,signal,self.analyses,samples,pre_scaled_pars='nuis',log_tag=log_tag)
        return q, joint_fitted, nuis_pars

    def fit_nuisance_and_scale(self,signal,samples,log_tag=''):
        """Fit nuisance parameters plus a signal scaling parameter
           (ignores parameters that were used to construct this object)"""
        pars = self.get_nuis_parameters(signal,samples)
        # Signal scaling parameter. One per sample, and per signal input
        Nsamples = list(samples.values())[0].shape[0]
        Nsignals = list(list(signal.values())[0].values())[0].shape[0] 
        muV = tf.Variable(np.zeros((Nsamples,Nsignals,1)),dtype=float)
        # Function to produce signal parameters from mu
        def mu_to_sig(pars):
            mu = tf.sinh(pars['mu']) # sinh^-1 is kind of like log, but stretches similarly for negative values. Seems to be pretty good for this.
            sig_out = {}
            for ka,a in signal.items():
                sig_out[ka] = {}
                for kp,p in a.items():
                    sig_out[ka][kp] = mu*p
            nuis_pars = {k:v for k,v in pars.items() if k is not 'mu'}
            out = deep_merge(nuis_pars,sig_out) # Return all signal and nuisance parameters, but not mu.
            return out 
        pars['mu'] = muV # TODO: Not attached to any analysis, but might work anyway  
        joint_fitted, q = optimize(pars,None,self.analyses,samples,pre_scaled_pars='nuis',transform=mu_to_sig,log_tag=log_tag)
        return q, joint_fitted, pars
  
    def fit_all(self,samples,log_tag=''):
        """Fit all signal and nuisance parameters to samples
           (ignores parameters that were used to construct this object)"""
        all_pars = self.get_all_parameters(samples)
        joint_fitted, q = optimize(all_pars,None,self.analyses,samples,pre_scaled_pars='all',log_tag=log_tag)
        return q, joint_fitted, all_pars

    def cat_pars(self,pars):
        """Stack tensorflow parameters in known order"""
        parlist = []
        maxdims = {}
        for ka,a in pars.items():
            for kp,p in a.items():
                parlist += [p]
                i = -1
                for d in p.shape[::-1]:
                    if i not in maxdims.keys() or maxdims[i]<d: maxdims[i] = d
                    i-=1
        maxshape = [None for i in range(len(maxdims))]
        for i,d in maxdims.items():
            maxshape[i] = d

        # Attempt to broadcast all inputs to same shape
        matched_parlist = []
        bcast = tf.broadcast_to(tf.constant(np.ones([1 for d in range(len(maxdims))]),dtype=float),maxshape)
        for p in parlist:
            matched_parlist += [p*bcast]
        return tf.Variable(tf.concat(matched_parlist,axis=-1),name="all_parameters")               

    def uncat_pars(self,catted_pars,pars_template=None):
        """De-stack tensorflow parameters back into separate variables of
           shapes know to each analysis. Assumes stacked_pars are of the
           same structure as pars_template"""
        if pars_template is None: pars_template = self.pars
        pars = {}
        i = 0
        for ka,a in pars_template.items():
            pars[ka] = {}
            for kp,p in a.items():
                N = p.shape[-1]
                pars[ka][kp] = catted_pars[...,i:i+N]
                i+=N
        return pars

    def Hessian(self,pars,samples):
        """Obtain Hessian matrix (and grad) at input parameter point
           Make sure to use de-scaled parameters as input!"""

        # Stack current parameter values to single tensorflow variable for
        # easier matrix manipulation
        catted_pars = self.cat_pars(pars)
        #print("catted_pats:", catted_pars)
        with tf.GradientTape(persistent=True) as tape:
            inpars = self.uncat_pars(catted_pars) # need to unstack for use in each analysis
            joint = JMCJoint(self.analyses,inpars,pre_scaled_pars=None)
            q = -2*joint.log_prob(samples)
            grads = tape.gradient(q, catted_pars)
        #print("grads:", grads)
        hessians = tape.batch_jacobian(grads, catted_pars) # batch_jacobian takes first (the sample) dimensions as independent for much better efficiency
        #print("H:",hessians)
        # Remove the singleton dimensions. We should not be computing Hessians for batches of signal hypotheses. TODO: throw better error if dimension sizes not 1
        grads_out = tf.squeeze(grads,axis=[-2])
        hessians_out = tf.squeeze(hessians,axis=[-2,-4])
        #print("g_out:",grads_out)
        #print("H_out:",hessians_out)
        return hessians_out, grads_out

    def decomposed_parameters(self,pars):
        """Separate input parameters into 'interest' and 'nuisance' lists,
           keeping tracking of their original 'indices' w.r.t. catted format.
           Mainly used for decomposing Hessian matrix."""
        interest_i = {} # indices of parameters in Hessian/stacked pars
        nuisance_i = {}
        interest_p = {} # parameters themselves
        nuisance_p = {}         
        i = 0
        for ka,a in pars.items():
            interest_p[ka] = {}
            nuisance_p[ka] = {}
            interest_i[ka] = {}
            nuisance_i[ka] = {}
            for kp,p in a.items():
                N = p.shape[-1]
                if kp=='theta': #TODO need a more general method for determining which are the nuisance parameters
                    nuisance_p[ka][kp] = p
                    nuisance_i[ka][kp] = (i, N)
                else:
                    interest_p[ka][kp] = p
                    interest_i[ka][kp] = (i, N)
                i+=N
        return interest_i, interest_p, nuisance_i, nuisance_p

    def sub_Hessian(self,H,pari,parj,idim=-1,jdim=-2):
        """Extract sub-Hessian matrix from full Hessian H,
           using dictionary that provides indices for selected
           parameters in H"""
        ilist = []
        for ai in pari.values():
            for i,Ni in ai.values():
                ilist += [ix for ix in range(i,i+Ni)] 
        jlist = []
        for aj in parj.values():
            for j,Nj in aj.values():
                jlist += [jx for jx in range(j,j+Nj)] 
        #print("H.shape:",H.shape)
        #print("ilist:",ilist)
        #print("jlist:",jlist)
        # Use gather to extract row/column slices from Hessian
        subH_i = tf.gather(H,      ilist, axis=idim)
        subH   = tf.gather(subH_i, jlist, axis=jdim)
        return subH

    def sub_grad(self,grad,pari,idim=-1):
        """Extract sub-gradient vector from full grad,
           using dictionary that provides indices for selected
           parameters in grad"""
        ilist = []
        for ai in pari.values():
            for i,Ni in ai.values():
                ilist += [ix for ix in range(i,i+Ni)] 
        sub_grad = tf.gather(grad, ilist, axis=idim)
        return sub_grad

    def decompose_Hessian(self,H,parsi,parsj):
        """Decompose Hessian matrix into
           parameter blocks"""
        Hii = self.sub_Hessian(H,parsi,parsi)
        Hjj = self.sub_Hessian(H,parsj,parsj)
        Hij = self.sub_Hessian(H,parsi,parsj) #Off-diagonal block. Symmetric so we don't need both.
        return Hii, Hjj, Hij

    def quad_loglike_prep(self,samples):
        """Compute second-order Taylor expansion of log-likelihood surface
           around input parameter point(s), and compute quantities needed
           for analytic determination of profile likelihood for fixed signal
           parameters, under this approximation."""
        pars = self.descale_pars(self.pars) # Make sure to use non-scaled parameters to get correct gradients etc.
        print("Computing Hessian and various matrix operations for all samples...")
        H, g = self.Hessian(pars,samples)
        #print("g:", g) # Should be close to zero if fits worked correctly
        interest_i, interest_p, nuisance_i, nuisance_p = self.decomposed_parameters(pars)
        #print("self.pars:", self.pars)
        #print("descaled_pars:", pars)
        #print("samples:", samples)
        #print("interest_p:", interest_p)
        #print("nuisance_p:", nuisance_p)
        Hii, Hnn, Hin = self.decompose_Hessian(H,interest_i,nuisance_i)
        Hnn_inv = tf.linalg.inv(Hnn)
        gn = self.sub_grad(g,nuisance_i)
        A = tf.linalg.matvec(Hnn_inv,gn)
        B = tf.linalg.matmul(Hnn_inv,Hin) #,transpose_b=True) # TODO: Not sure if transpose needed here. Doesn't seem to make a difference, which seems a little odd.
        print("...done!")
        return A, B, interest_p, nuisance_p

    def quad_loglike_f(self,samples):
        """Return a function that can be used to compute the profile log-likelihood
           for fixed signal parametes, for many signal hypotheses, using a 
           second-order Taylor expandion of the likelihood surface about a point to
           determine the profiled nuisance parameter values. 
           Should be used after pars are fitted to the global best fit for best
           expansion."""
        A, B, interest_p, nuisance_p = self.quad_loglike_prep(samples)
        f = mm.tools.func_partial(self.neg2loglike_quad,A=A,B=B,interest=interest_p,nuisance=nuisance_p,samples=samples)
        return f

    def neg2loglike_quad(self,signal,A,B,interest,nuisance,samples):
        """Compute -2*loglikelihood using pre-computed Taylor expansion
           parameters (for many samples) for a set of signal hypotheses"""
        # Stack signal parameters into appropriate vector for matrix operations
        parlist = []
        for ka,a in interest.items():
            if ka not in signal.keys():
                raise ValueError("No test signals provided for analysis {0}".format(ka)) 
            for kp in a.keys():
                if kp not in signal[ka].keys():
                    raise ValueError("No test signals provided for region {0} in analysis {1}".format(kp,ka))
                parlist += [signal[ka][kp]]
        s = tf.constant(tf.concat(parlist,axis=-1),name="all_signal_parameters")
        s_0 = self.cat_pars(interest) # stacked interest parameter values at expansion point
        theta_0 = self.cat_pars(nuisance) # stacked nuisance parameter values at expansion point
        #print("theta_0.shape:",theta_0.shape)
        #print("A.shape:",A.shape)
        #print("B.shape:",B.shape)
        #print("s.shape:",s.shape)
        #print("s_0.shape:",s_0.shape)
        theta_prof = theta_0 - tf.expand_dims(A,axis=1) - tf.linalg.matvec(tf.expand_dims(B,axis=1),tf.expand_dims(s,axis=0)-s_0)
        #theta_prof = theta_0 - tf.linalg.matvec(tf.expand_dims(B,axis=1),tf.expand_dims(s,axis=0)-s_0) # Ignoring grad term
        # de-stack theta_prof
        theta_prof_dict = self.uncat_pars(theta_prof,pars_template=nuisance)
        #print("theta_prof_dict:", theta_prof_dict)
        #print("signal:", signal)
        # Compute -2*log_prop
        joint = JMCJoint(self.analyses,deep_merge(signal,theta_prof_dict),pre_scaled_pars=None)
        q = -2*joint.log_prob(samples)
        return q

def LEEcorrection(analyses,signal,nosignal,name,N,fitall=True):
    """Compute LEE-corrected p-value to exclude the no-signal hypothesis, with
       'signal' providing the set of signals to consider for the correction
       
       Also computes local p-values for all input signal hypotheses.
    """
    nullnuis = {a.name: {'nuisance': None} for a in analyses.values()} # Use to automatically set nuisance parameters to zero for sample generation

    print("Simulating {0}".format(name))
    
    # Create joint distributions
    joint   = JMCJoint(analyses) # Version for fitting (object is left with fitted parameters upon fitting)
    joint0  = JMCJoint(analyses,deep_merge(nosignal,nullnuis))
    joint0s = JMCJoint(analyses,deep_merge(signal,nullnuis))
    
    # Get Asimov samples for nuisance MLEs with fixed signal hypotheses
    samplesAb = joint0.Asamples
    samplesAsb = joint0s.Asamples 
 
    # Get observed data
    obs_data = joint0.Osamples
    
    # Evaluate distributions for Asimov datasets, in case where we
    # know that the MLEs for those samples are the true parameters
    qsbAsb = -2*(joint0s.log_prob(samplesAsb))
    qbAb  = -2*(joint0.log_prob(samplesAb))
    
    # Fit distributions for Asimov datasets for the other half of each
    # likelihood ratio
    print("Fitting w.r.t Asimov samples")
    qbAsb, joint_fitted, pars = joint.fit_nuisance(nosignal, samplesAsb,log_tag='bAsb')
    qsbAb, joint_fitted, pars = joint.fit_nuisance(signal, samplesAb,log_tag='sbAsb')
  
    qAsb = (qsbAsb - qbAsb)[0] # extract single sample result
    qAb = (qsbAb - qbAb)[0]
    
    # Generate background-only pseudodata to be fitted
    samples0 = joint0.sample(N)
    onesample0 = joint0.sample(1) # For some quick stuff
     
    # Generate signal pseudodata to be fitted
    samples0s = joint0s.sample(N)
   
    if fitall:
        print("Fitting GOF w.r.t background-only samples")
        qgof_b, joint_gof_fitted_b, gof_pars_b  = joint.fit_all(samples0, log_tag='gof_all_b')
        #print("Fitting GOF w.r.t signal samples")
        #qgof_sb, joint_gof_fitted_sb ,gof_pars_sb  = joint.fit_all(samples0s, log_tag='gof_all_sb')

        print("Fitting w.r.t background-only samples")
        qb , joint_fitted_b, nuis_pars_b = joint.fit_nuisance(nosignal, samples0, log_tag='qb')
        qsb, joint_fitted_sb, nuis_pars_s = joint.fit_nuisance(signal, samples0, log_tag='qsb')
        q = qsb - qb # mu=0 distribution
    else:
        # Only need the no-signal nuisance parameter fits for quadratic approximations
        print("Fitting no-signal nuisance parameters w.r.t background-only samples")
        qb, joint_fitted_b, nuis_pars_b = joint.fit_nuisance(nosignal, samples0, log_tag='qb')

        # Do one full GOF fit just to determine parameter numbers 
        null, null, gof_pars_b  = joint.fit_all(onesample0)

    # Obtain function to compute neg2logl for fitted samples, for any fixed input signal,
    # with nuisance parameters analytically profiled out using a second order Taylor expansion
    # about the GOF best fit points.
    #print("fitted pars:", gof_pars_b)
    #f1 = joint_gof_fitted_b.quad_loglike_f(samples0)

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

    # print("Fitting w.r.t signal samples")
    # qb_s , joint_fitted, nuis_pars = joint.fit_nuisance(nosignal, samples0s)
    # qsb_s, joint_fitted, nuis_pars = joint.fit_nuisance(signal, samples0s)
    # q_s = qsb_s - qb_s #mu=1 distribution

    print("Determining all local p-value distributions...")
    #for i in range(q_quad.shape[-1]):
    #    if (i%1000)==0: print("   {0} of {1}".format(i,q_quad.shape[-1]))
    #    if fitall: 
    #        #MCcdf, order = CDFf(q[:,i].numpy())
    #        #pval = MCcdf(q[:,i])
    #        pval = eCDF(q[:,i].numpy())
    #        fullMCp = -sps.norm.ppf(pval)
    #        plocal     += [pval]
    #        sigmalocal += [fullMCp]
    #    #MCcdf_quad, order = CDFf(q_quad[:,i].numpy())
    #    #pval_quad = MCcdf_quad(q_quad[:,i])
    #    pval_quad = eCDF(q_quad[:,i].numpy())
    #    quadMCp = -sps.norm.ppf(pval_quad)
    #    plocal_quad += [pval_quad]
    #    sigmalocal_quad += [quadMCp]

    #if fitall: 
    #    pval = eCDF(tf.sort(q,axis=0))
    #    fullMCp = -sps.norm.ppf(pval)
    #    plocal     += [pval]
    #    sigmalocal += [fullMCp]
    #    pval = eCDF(tf.sort(q,axis=0))
    q_quad_sort_i = tf.argsort(q_quad,axis=0)
    cdf = eCDF(q_quad)
    # Need to undo the sort to assign p-values back to where they belong
    unsort_i = tf.argsort(q_quad_sort_i,axis=0)
    plocal_all_quad     = tf.gather(cdf, unsort_i)
    sigmalocal_all_quad = -sps.norm.ppf(plocal_all_quad)

    # Select minimum p-values for each sample from across all signal hypotheses (tests)
    if fitall:
        #plocal_all     = tf.stack(plocal,axis=-1)
        #sigmalocal_all = tf.stack(sigmalocal,axis=-1)
        q_min_i = tf.argmin(q,axis=-1) # "Best fit" hypothesis
        #sigmalocal_min_i      = tf.argmax(sigmalocal_all,axis=-1) # Could also select based on exclusion of b-only hypothesis. But a bit weird to do that.
        q_min     = gather_by_idx(q,q_min_i)
   
        # Local p-values at selected point
        plocal_BF          = gather_by_idx(plocal_all,q_min_i).numpy()    
        sigmalocal_BF      = gather_by_idx(sigmalocal_all,q_min_i).numpy()

    #plocal_all_quad = tf.stack(plocal_quad,axis=-1)
    #sigmalocal_all_quad = tf.stack(sigmalocal_quad,axis=-1)
    #sigmalocal_min_quad_i = tf.argmax(sigmalocal_all_quad,axis=-1)
    q_min_quad_i = tf.argmin(q_quad,axis=-1)
    qquad_min = gather_by_idx(q_quad,q_min_quad_i) 
    sigmalocal_B_quad  = gather_by_idx(sigmalocal_all_quad,q_min_quad_i).numpy()
    plocal_BF_quad     = gather_by_idx(plocal_all_quad,q_min_quad_i).numpy()
  
    # GOF distributions
    if fitall:
        qgofb_true = qb - qgof_b # Test of b-only when b is true
        #qgofsb_true = qsb_s - qgof_sb # Test of s when s is true
        # the above should both be asymptotically chi^2
    N_gof_pars = sum([par.shape[-1] for a in gof_pars_b.values() for par in a.values()])
    N_nuis_pars = sum([par.shape[-1] for a in nuis_pars_b.values() for par in a.values()])
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
    
    #print("GOF BF pars:", pars)

    # Plots!
    # First: GOF b-only samples, vs selected lowest p-value
    #  I think in ideal cases should be the same.
    fig  = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax2.set(yscale="log")
    
    if fitall:
        sns.distplot(qgofb_true, color='b', kde=False, ax=ax1, norm_hist=True, label="GOF MC")
        sns.distplot(qgofb_true, color='b', kde=False, ax=ax2, norm_hist=True, label="GOF MC")
        sns.distplot(-q_min, color='g', kde=False, ax=ax1, norm_hist=True, label="LEEC")
        sns.distplot(-q_min, color='g', kde=False, ax=ax2, norm_hist=True, label="LEEC")
    sns.distplot(-qquad_min, color='m', kde=False, ax=ax1, norm_hist=True, label="LEEC quad")
    sns.distplot(-qquad_min, color='m', kde=False, ax=ax2, norm_hist=True, label="LEEC quad")
   
    qx = np.linspace(0, np.max(-qquad_min),1000) # 6 sigma too far for tf, cdf is 1. single-precision float I guess
    qy = tf.math.exp(tfd.Chi2(df=DOF).log_prob(qx))
    sns.lineplot(qx,qy,color='g',ax=ax1, label="asymptotic")
    sns.lineplot(qx,qy,color='g',ax=ax2, label="asymptotic")

    ax1.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
    ax2.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
   
    fig.tight_layout()
    fig.savefig("qGOF_dists_v_leeC_{0}.png".format(name))

    if fitall:
        fig3  = plt.figure(figsize=(6,4))
        ax31 = fig3.add_subplot(111)
        diag = np.min(fullMCp), np.max(fullMCp)
        print("diag:",diag)
        ax31.plot(diag,diag,c='k')
        ax31.scatter(fullMCp,quadMCp,s=1,c='b')
        ax31.set_xlabel("full MC sigma")
        ax31.set_ylabel("quad. approx sigma")
        fig3.savefig("qb_vs_qquad_last_{0}.png".format(name))

    # Distribution of local p-value at best-fit point and
    # local p-value at best-fit point VS LEE corrected p-value

    fig4  = plt.figure(figsize=(12,4))
    ax1 = fig4.add_subplot(1,2,1)
    ax2 = fig4.add_subplot(1,2,2)
    ax1.set(xscale="log")
    ax1.set(yscale="log")

    if fitall:
        pcdf,      order = CDFf(plocal_BF)
        LEEsigma      = -sps.norm.ppf(pcdf(plocal_BF))
        # Predictions of asymptotic theory (for the full signal parameter space)
        p_asympt = 1 - sps.chi2(df=DOF).cdf(-q_min)
        sigma_asympt = -sps.norm.ppf(p_asympt)
    pcdf_quad, order_quad = CDFf(plocal_BF_quad)
    LEEsigma_quad = -sps.norm.ppf(pcdf_quad(plocal_BF_quad))

    # BF local p-value distribtuion
    diag = np.min(plocal_BF_quad), np.max(plocal_BF_quad)
    ax1.plot(diag,diag,c='k')
    if fitall:
        #ax1.plot(plocal_BF[order],p_asympt[order],drawstyle='steps-post',label='Asympt.',c='k',alpha=0.6)
        ax1.scatter(plocal_BF[order],p_asympt[order],s=1.5,lw=0,label='Asympt.',c='k',alpha=0.6)
        ax1.plot(plocal_BF[order],pcdf(plocal_BF[order]),drawstyle='steps-post',label='Full MC',c='b')
    ax1.plot(plocal_BF_quad[order_quad],pcdf_quad(plocal_BF_quad[order_quad]),drawstyle='steps-post',label='Quad approx.',c='m')
    ax1.set_xlabel("local p-value")
    ax1.set_ylabel("CDF")
    ax1.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
 
    # BF local sigma vs LEE-corrected sigma (basically same as above, just different scale)
    diag = np.min(-sps.norm.ppf(plocal_BF_quad)), np.max(-sps.norm.ppf(plocal_BF_quad))
    ax2.plot(diag,diag,c='k')
    if fitall:
        #ax2.plot(-sps.norm.ppf(plocal_BF[order]),sigma_asympt[order],drawstyle='steps-post',label='Asympt.',c='k',alpha=0.6)
        ax2.scatter(-sps.norm.ppf(plocal_BF[order]),sigma_asympt[order],s=1.5,lw=0,label='Asympt.',c='k',alpha=0.6)
        ax2.plot(-sps.norm.ppf(plocal_BF[order]),LEEsigma[order],drawstyle='steps-post',label='Full MC',c='b')
    ax2.plot(-sps.norm.ppf(plocal_BF_quad[order_quad]),LEEsigma_quad[order_quad],drawstyle='steps-post',label='Quad approx.',c='m')
    ax2.set_xlabel("local sigma")
    ax2.set_ylabel("global sigma")
    ax2.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
 
    fig4.savefig("local_vs_global_sigma_{0}.png".format(name))

class LEECorrectorMaster:
    """A class to wrap up analysis and database access routines to manage LEE corrections
       for large numbers of analyses, with many signal hypotheses to test, and with many
       random MC draws. Allows more signal hypotheses and random draws to be added to the
       simulation without recomputing everything"""

    def __init__(self,analyses,db):
        self.analyses = analyses

    def add_signals(self,signals):
        pass

    def add_to_database(self):
        pass

    def compute_quad(self,data):
        """Compute quadratic approximations of profile likelihood for a set of
           data"""
        pass

def sql_create_table(c,table_name,columns):
    """Create an SQLite table if it doesn't already exist"""
    command = "CREATE TABLE IF NOT EXISTS {0} ({1} {2}".format(table_name,columns[0][0],columns[0][1])
    for col, t in columns[1:]:
        command += ",{0} {1}".format(col,t)
    command += ")"  
    #print("command:",command)
    c.execute(command)

def sql_upsert(c,table_name,df,primary):
    """Insert or overwrite data into a set of SQL columns.
       Data assumed to be a pandas dataframe. Must specify
       which column contains the primary key.
    """
    rec = df.to_records()
    print("rec:",rec)
    columns = df.to_records().dtype.names  
    command = "INSERT INTO {0} ({1}".format(table_name,columns[0])
    if len(columns)>1:
        for col in columns[1:]:
            command += ",{0}".format(col)
    command += ") VALUES (?"
    if len(columns)>1:
        for col in columns[1:]:
            command += ",?"
    command += ")"
    if len(columns)>1:
        command += " ON CONFLICT({0}) DO UPDATE SET ".format(primary)
        for j,col in enumerate(columns):
            if col is not primary:
                command += "{0}=excluded.{0}".format(col)
                if j<len(columns)-1: command += ","
    print("command:", command)
    c.executemany(command,  map(tuple, rec.tolist())) # sqlite3 doesn't understand numpy types, so need to convert to standard list. Seems fast enough though.

def sql_load(c,table_name,keys,primary,cols):
    """Load data from an sql table
       with simple selection of items by primary key values.
       Compressed primary key values into a set of ranges to
       construct more efficient queries.
       Assumes the primary key values are supplied 
       in ascending order."""

    splitdata = np.split(keys, np.where(np.diff(keys) != 1)[0]+1)
    ranges = [(np.min(x), np.max(x)) for x in splitdata]

    command = "SELECT "
    for col in cols:
        command += cols+","
    command = command[:-1]
    command += " from events WHERE "
    for i,(start, stop) in enumerate(ranges):
        command += " {0} BETWEEN {1} and {2}".format(primary,start,stop) # inclusive 'between' 
        if i<len(ranges)-1: command += " OR "
    c.execute(command)
    return c.fetchall() 
 
class LEECorrectorAnalysis:
    """A class to wrap up a SINGLE analysis with connection to output sql database,
       to be used as part of look-elsewhere correction calculations

       The structure of output is as follows:
        - 1 master directory for each complete analysis to be performed, containing:
          - 1 database file per Analysis class, containing:
            - 1 'master' table assigning IDs (just row numbers) to each 'event', or pseudoexperiment, and recording the data
            - 1 table synchronised with the 'master' table, containing local test statistic values associated with the (fixed) observed 'best fit' signal point
              (may add more for other 'special' signal points for which we want local test statistic stuff)
            - 1 table  "                            "                        test statistic values associated with the combined best fit point for each event
          - 1 datafiles file for the combination, containing:
            - 1 table containining combined test statistic values for each event (synchronised with each Analysis file)

        We have to compute test statistic values for *ALL* signal hypotheses to be considered, for every event, however
        this is too much data to store. So we keep only the 'profiled' test statistic values, i.e. the values extremised
        over the available signal hypotheses. If more signal hypotheses are added, we can just compare to the already-computed
        extrema test statistic values for each event to see if they need to be updated.
    """

    def __init__(self,analysis,path,comb_name,nullsignal):
        self.event_table = 'events'
        self.background_table = 'background'
        self.combined_table = comb_name # May vary, so that slightly different combinations can be done side-by-side
        self.analysis = analysis
        self.analyses = {self.analysis.name: self.analysis} # For functions that expect a dictionary of analyses
        self.nullsignal = nullsignal # "signal" parameters to be used as the 'background-only' null hypothesis
        self.nullnuis = {self.analysis.name: {'nuisance': None}} # Use to automatically set nuisance parameters to zero for sample generation
        pathlib.Path(path).mkdir(parents=True, exist_ok=True) # Ensure output path exists
        self.db = '{0}/{1}.db'.format(path,self.analysis.name)
        self.joint = JMCJoint(self.analyses)
        conn = self.connect_to_db()

        # Columns for pseudodata table
        self.event_columns = ["EventID"] # ID number for event, used as index for table  
        for name,dim in self.analysis.get_sample_structure().items():
           self.event_columns += ["{0}_{1}".format(name,i) for i in range(dim)]

        #print("columns:", self.event_columns)

        c = conn.cursor()
        cols = [("EventID", "integer primary key")]
        cols += [(col, "real") for col in self.event_columns[1:]]
        sql_create_table(c,self.event_table,cols) 
        self.check_table(c,self.event_table,self.event_columns)
       
        # Table for background-only fit data
        test_stat_cols = [("EventID", "integer primary key")]
        test_stat_cols += [(col, "real") for col in ["neg2logL"]]
        colnames = [x[0] for x in test_stat_cols]
 
        # Will also need nuisance parameter fit values:
        nuis_cols = []
        nuis_structure = self.analysis.get_nuisance_parameter_structure()
        for par,size in nuis_structure.items():
            for i in range(size):
                nuis_cols += [("{0}_{1}".format(par,i), "real")]

        bg_cols = test_stat_cols + nuis_cols
        bg_colnames = [x[0] for x in bg_cols]
        sql_create_table(c,self.background_table,bg_cols) 
        self.check_table(c,self.background_table,bg_colnames)

        # Table for best-fit (combined) signal model fit data
        comb_cols = test_stat_cols + [("neg2logL_quad", "real")]
        comb_colnames = [x[0] for x in comb_cols]
        sql_create_table(c,self.combined_table,comb_cols) 
        self.check_table(c,self.combined_table,comb_colnames)

        self.close_db(conn)

    def connect_to_db(self):
        conn = sqlite3.connect(self.db) 
        return conn

    def close_db(self,conn):
        conn.commit()
        conn.close()

    def check_table(self,c,table,required_cols):
        # Print the columns already existing in our table
        c.execute('PRAGMA table_info({0})'.format(table))
        results = c.fetchall()
        existing_cols = [row[1] for row in results]

        # Check if they match the required columns
        if existing_cols != required_cols:
            msg = "Existing table '{0}' for analyis {1} does not contain the expected columns! May have been created with a different version of this code. Please delete it to recreate the table from scratch.\nExpected columns: {2}\nActual columns:{3}".format(table,self.analysis.name,required_cols,existing_cols)
            raise ValueError(msg) 

    def add_events(self,signal,N):
        """Add N new events generated under 'signal' to the event table"""
        print("Recording {0} new events...".format(N))
        start = time.time()
        joint = JMCJoint(self.analyses,deep_merge(signal,self.nullnuis))

        # Generate pseudodata
        samples = joint.sample(N)

        # Save events to database
        conn = self.connect_to_db()
        c = conn.cursor()
   
        structure = self.analysis.get_sample_structure() 
        
        command = "INSERT INTO 'events' ("
        for name, size in structure.items():
            for j in range(size):
                command += "'{0}_{1}',".format(name,j)
        command = command[:-1] # remove trailing comma
        command += ") VALUES ("
        for name, size in structure.items():
            for j in range(size):
                command += "?," # Data provided as second argument to 'execute'
        command = command[:-1] # remove trailing comma
        command += ")"         
            
        # Paste together data tables
        datalist = []
        for name, size in structure.items():
            datalist += [tf.squeeze(samples[self.analysis.name+"::"+name],axis=1)] # Remove 'signal' dimension, should be size 1 (TODO: add check for this)
         
        datatable = tf.concat(datalist,axis=-1).numpy()

        c.executemany(command,  map(tuple, datatable.tolist())) # sqlite3 doesn't understand numpy types, so need to convert to standard list. Seems fast enough though.
        self.close_db(conn)
        end = time.time()
        print("Took {0} seconds".format(end-start))

    def load_events(self,N,reftable,condition,offset=0):
        """Loads N events from database where 'condition' is true in 'reftable'
           To skip rows, set 'offset' to the first row to be considered."""
        structure = self.analysis.get_sample_structure() 
        conn = self.connect_to_db()
        c = conn.cursor()

        # First see if any data is in the reference table yet:
        c.execute("SELECT Count(EventID) FROM "+reftable)
        results = c.fetchall()

        nrows = results[0][0]
        command = "SELECT A.EventID"
        for name, size in structure.items():
            for j in range(size):
                command += ",A.{0}_{1}".format(name,j)

        command += " from events as A"        
        if nrows!=0:
           # Apply extra condition to get only events where "neg2LogL" column is NULL (or the EventID doesn't exist) in the 'background' table
           command += """
                      left outer join {0} as B
                          on A.EventID=B.EventID
                      where
                          B.{1} 
                      """.format(reftable,condition)
        command += " LIMIT {0} OFFSET {1}".format(N,offset)
        c.execute(command)
        results = c.fetchall()
        self.close_db(conn) 

        if len(results)>0:
            # Convert back into dictionary of tensorflow tensors
            # Start by converting to one big tensor
            alldata = tf.convert_to_tensor(results, dtype=tf.float32)
 
            EventIDs = tf.cast(tf.round(alldata[:,0]),dtype=tf.int32) # TODO: is this safe?
            i = 1;
            events = {}
            for name, size in structure.items():
                subevents = tf.expand_dims(alldata[:,i:i+size],axis=1) # insert the 'signal' parameter dimension
                i+=size
                events[self.analysis.name+"::"+name] = subevents

            print("Retreived eventIDs {0} to {1}".format(np.min(EventIDs.numpy()),np.max(EventIDs.numpy())))
        else:
            EventIDs = None
            events = None
        return EventIDs, events

    def load_eventIDs(self,EventIDs):
        """Loads events with the given eventIDs"""
        structure = self.analysis.get_sample_structure() 
        conn = self.connect_to_db()
        c = conn.cursor()
 
        cols = []
        for name, size in structure.items():
            for j in range(size):
                cols += ["{0}_{1}".format(name,j)]

        results = sql_load(c,'events',EventIDs,'EventID',cols)

        c.execute(command)
        results = c.fetchall()
        self.close_db(conn) 

        if len(results)>0:
            # Convert back into dictionary of tensorflow tensors
            # Start by converting to one big tensor
            alldata = tf.convert_to_tensor(results, dtype=tf.float32)
            i = 0;
            events = {}
            for name, size in structure.items():
                subevents = tf.expand_dims(alldata[:,i:i+size],axis=1) # insert the 'signal' parameter dimension
                i+=size
                events[self.analysis.name+"::"+name] = subevents
        else:
            events = None
        return events

    def load_bg_nuis_pars(self,EventIDs):
        """Loads fitted background-only nuisance parameter values for the selected
           events"""
        nuis_structure = self.analysis.get_nuisance_parameter_structure()
        cols = []
        for par,size in nuis_structure.items():
            for i in range(size):
                cols += ["{0}_{1}".format(par,i)]

        conn = self.connect_to_db()
        c = conn.cursor()

        results = sql_load(c,'background',EventIDs,'EventID',cols)
        print("bg_pars:", results)

        # TODO: convert back to tensor
        # if len(results)>0:
        #     # Convert back into dictionary of tensorflow tensors
        #     # Start by converting to one big tensor
        #     alldata = tf.convert_to_tensor(results, dtype=tf.float32)
        #     i = 0;
        #     events = {}
        #     for name, size in structure.items():
        #         subevents = tf.expand_dims(alldata[:,i:i+size],axis=1) # insert the 'signal' parameter dimension
        #         i+=size
        #         events[self.analysis.name+"::"+name] = subevents
        # else:
        #     events = None
        # return events

    def fit_signal_batch(self,events,signals):
        """Compute signal fits for selected eventIDs
           Returns test statistic values for combination
           with other analyses. No results recorded; this
           needs to be request in a separate step by the calling
           code, by running 'record_bf_signal_stats'
        """
        # Run full numerical fits of nuisance parameters for all signal hypotheses
        qsb, joint_fitted_sb, nuis_pars_s = self.joint.fit_nuisance(signals, events, log_tag='qsb')
        return qsb

        # Estimate profile likelihood using quadratic approximation in generalised signal parameter space.
        f2 = joint_fitted_b.quad_loglike_f(events)
        #qsb_quad = f2(signals) # calling code can do this themselves, and thus only call this function once per batch of events, however many signals there are to consider. Saves re-doing the matrix calculations associated with computing the quadratic approximation.

        return qsb, f2

    def compute_quad(self,events):
        """Compute quadratic approximations of profile likelihood for the specified
           set of events"""
        f2 = joint_fitted_b.quad_loglike_f(events)
        pass

    def record_bf_signal_stats(self):
        pass

    def process_background(self):
        """Compute background-only fits for events currently in our output tables
           But only for events where this hasn't already been done."""

        joint = JMCJoint(self.analyses)

        batch_size = 10000
        continue_processing = True
        while continue_processing:
            EventIDs, events = self.load_events(batch_size,'background','neg2logL is NULL')
            if EventIDs is None:
                # No events left to process
                continue_processing = False
            if continue_processing:
                print("Fitting w.r.t background-only samples")
                qb, joint_fitted_b, nuis_pars_b = joint.fit_nuisance(self.nullsignal, events, log_tag='qb')
                # Write events to output database               
                # Write fitted nuisance parameters to disk as well, for later use in constructing quadratic approximation of profile likelihood
                arrays = [qb]
                cols = ["neg2logL"]
                for par, arr in nuis_pars_b[self.analysis.name].items():
                    for i in range(arr.shape[-1]):
                        cols += ["{0}_{1}".format(par,i)]
                    arrays += [tf.squeeze(arr,axis=1)] # remove 'signal' dimension
                allpars = tf.concat(arrays,axis=-1)               
                data = pd.DataFrame(allpars.numpy(),index=EventIDs.numpy(),columns=cols)
                data.index.name = 'EventID' 
                conn = self.connect_to_db()
                c = conn.cursor()
                sql_upsert(c,'background',data,primary='EventID')
                self.close_db(conn)

def collider_analyses_from_long_YAML(yamlfile,replace_SR_names=False):
    """Read long format YAML file of analysis data into ColliderAnalysis objects"""
    #print("Loading YAML")
    d = yaml.safe_load(yamlfile)
    #print("YAML loaded")
    analyses = {}
    SR_name_translation = {}
    inverse_translation = {}
    nextID=0
    for k,v in d.items():
        if replace_SR_names:
            # TensorFlow cannot handle some characters, so switch SR names to something simple
            SR_name_translation.update({"SR{0}".format(nextID+i): sr[0] for i,sr in enumerate(v["Signal regions"])})
            inverse_translation.update({sr[0]: "SR{0}".format(nextID+i) for i,sr in enumerate(v["Signal regions"])})
            srs = [["SR{0}".format(nextID+i)]+sr[1:] for i,sr in enumerate(v["Signal regions"])]
            nextID += len(srs)
        else:
            srs = v["Signal regions"]
        if "cov" in v.keys(): 
            cov = v["cov"]
            cov_order = v["cov_order"]
        else:
            cov = None
            cov_order = None
        ucz = v["unlisted_corr_zero"]
        a = ColliderAnalysis(k,srs,cov,cov_order,ucz)
        analyses[a.name] = a
    if replace_SR_names:
        return analyses, SR_name_translation, inverse_translation
    else:
        return analyses
