import numpy as np
import yaml
import pandas as pd
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import copy

# Stuff to help format YAML output
class blockseqtrue( list ): pass
def blockseqtrue_rep(dumper, data):
        return dumper.represent_sequence( u'tag:yaml.org,2002:seq', data, flow_style=True )
yaml.add_representer(blockseqtrue, blockseqtrue_rep)

# Want to convert all this to YAML. Write a simple container to help with this.
class ColliderAnalysis:
    def __init__(self,name,srs=None,cov=None,cov_order=None,unlisted_corr_zero=False):
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

    def tensorflow_model(self,signal,nuispars):
        """Output tensorflow probability model object, to be combined together and
           sampled from. Signal parameters should all be fixed."""

        # Need to construct these shapes to match the event_shape, batch_shape, sample_shape 
        # semantics of tensorflow_probability.

        cov_order = self.get_cov_order()
 
        tfds = []
        sample_count = 0 # Length of sample vector for this model
        sample_layout = [] # Record of structure of sample vector, so we can interpret it later
        for sr, bexp in zip(self.SR_names, self.SR_b):
            s = signal[sr]
            b = tf.constant(bexp,dtype=float)

            # Poisson model
            poises0  = tfd.Poisson(rate = s+b)
            tfds += [poises0]
            sample_layout += [(sr,"n",1)] # Signal region, sample name, sample vector length
            sample_count += 1

        # Put all the nuisance parameter sample at the end of the vector, so that it
        # is the same structure whether they come from a multivariate normal or multiple
        # indepdent normal distributions.
        for sr, bsysin in zip(self.SR_names, self.SR_b_sys):
            bsys = tf.constant(bsysin,dtype=float)
            #bsys_tmp = tf.expand_dims(tf.constant(bsysin,dtype=float),0) # Need to match shape of signal input
            #bsys = tf.broadcast_to(bsys_tmp,shape=signal[sr].shape)
            if self.cov is None or sr not in cov_order:
                # if no covariance info for this SR:
                #  Either 1. combine assuming independent, or 2. need to select one SR to use.
                #  For now just do 1. TODO: will need 2 also at some point
         
                # Nuisance parameters, use nominal null values (independent Gaussians to model background/control measurements)
                # Want to scale parameters so their 1 sigma fit width is about 1, to help out optimiser.
                theta = nuispars[sr] * bsys
                nuis0 = tfd.Normal(loc = theta, scale = bsys) # I forget if bsys includes the Poisson uncertainty on b TODO: Check this!
                tfds += [nuis0]
                sample_layout += [(sr,"x",1)]
                sample_count += 1

        if self.cov is not None:
            # Construct multivariate normal model for background/control measurements
            cov = tf.constant(self.cov,dtype=float)
            # Want to scale parameters so their 1 sigma fit width is about 1, to help out optimiser.
            theta_vec = tf.stack([nuispars[sr]*np.sqrt(self.cov[i][i]) for i,sr in enumerate(cov_order)],axis=-1) 
            #cov = tf.expand_dims(tf.constant(self.cov,dtype=float),0) # Need to match nuisance par shape (may be one for each of many signal hypotheses)
            #print("theta_vec.shape:",theta_vec.shape)
            #print("cov.shape:",cov.shape)
            cov_nuis = tfd.MultivariateNormalFullCovariance(loc=theta_vec,covariance_matrix=cov)
            tfds += [cov_nuis]
            sample_layout += [("cov","x",len(cov_order))]
            sample_count += 1 # Still counts as one item, since these samples will all come packaged together in the sample list

        return tfds, sample_layout, sample_count

    def tensorflow_null_model(self,signal):
        # Make sure signal consists of constant tensorflow objects
        sig = {sr: tf.constant(signal[sr], dtype=float) for sr in self.SR_names}
        zeros = {sr: tf.constant([0]*len(signal[sr]), dtype=float) for sr in self.SR_names}
        tfds, sample_layout, sample_count = self.tensorflow_model(sig,zeros)
        # Get Asimov samples while we are at it
        Asamples = self.get_Asimov_samples(sample_layout,signal)
        return tfds, sample_layout, sample_count, Asamples

    def samples_to_dict(self,samples,sample_layout):
        """Convert vector of samples into dictionary for easier usage"""
        sample_dict = {}
        i=0
        for name, v, count in sample_layout:
            if name=='cov':
                if v!='x':
                    raise ValueError("Variable named {0} encounted while parsing sample_layout item 'cov'. This should only contain 'x' variables! Something is wrong.")
                for j,sr in enumerate(self.get_cov_order()):
                    if sr not in sample_dict: sample_dict[sr] = {}
                    sample_dict[sr]['x'] = samples[i][...,j]
            else:
                if name not in sample_dict: sample_dict[name] = {}
                sample_dict[name][v] = samples[i]
            i+=count
        return sample_dict

    def get_Asimov_samples(self,sample_layout,signal):
        """Construct 'Asimov' samples for this analysis
           Used to detemine asymptotic distribtuions of 
           certain test statistics"""

        # Need to manually set Asimov samples for the model according to the sample layout
        Asamples = [];
        name_to_index = {sr: i for i,sr in enumerate(self.SR_names)}
        signal_len = len(signal[self.SR_names[0]]) # TODO: Should be the same for all SRs, could check this.
        for name, v, count in sample_layout:
            if name=='cov':
                if v!='x':
                    raise ValueError("Variable named {0} encounted while parsing sample_layout item 'cov'. This should only contain 'x' variables! Something is wrong.")
                cov_order = self.get_cov_order()
                Xsamp = np.zeros((signal_len,len(cov_order)))
                for j,sr in enumerate(self.get_cov_order()):
                    i = name_to_index[sr]
                    Xsamp[:,j] = 0 * signal[sr]
                Asamples += [Xsamp]
            else:
                i = name_to_index[name]
                if v=='n':
                    Asamples += [self.SR_b[i] + signal[name]] # signal region counts
                elif v=='x':
                    Asamples += [0 * signal[name]] # control measurements
        return Asamples 

    def get_tensorflow_variables(self,samples,sample_layout,signal):
        """Get parameters to be optimized, for input to "tensorflow_free_model"""
        # TODO: initial value (guess) for nuisance pars needs to be set carefully
        sample_dict = self.samples_to_dict(samples,sample_layout)
        seeds = self.get_seeds_nuis(sample_dict,signal)
        thetas = {sr: tf.Variable(seeds[sr]['theta'], dtype=float, name='theta_{0}'.format(sr)) for sr in self.SR_names}
        return thetas

    def tensorflow_free_model(self,signal,thetas):
        """Output tensorflow input parameter tensors and probability objects for this analysis,
           for optimisation/profiling relative to some simulated data
           Input:
              dictionary of signal parameter tensors to use
        """

        sig = {sr: tf.constant(signal[sr], dtype=float) for sr in self.SR_names}
        tfds, sample_layout, sample_count = self.tensorflow_model(sig,thetas)
        return tfds, sample_layout, sample_count 
       
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

    def get_seeds_nuis(self,samples,signal):
        """Get seeds for (additive) nuisance parameter fits,
           assuming fixed signal parameters. Gives exact MLEs in
           the absence of correlations
           TODO: switch off fits for exactly fitted parameters?
           """
        verbose = False # For debugging
        if verbose: print("signal (seeds_nuis):",signal)
        seeds={}
        self.theta_both={} # For debugging, need to compare BOTH solutions to numerical MLEs.
        self.theta_dat={}

        threshold = 1e-2 # Smallness threshold, for fixing numerical errors and disallowing solutions too close to zero

        for i,sr in enumerate(self.SR_names): 
            seeds[sr] = {}
            # From input
            n = samples[sr]['n'] 
            x = samples[sr]['x']
            s = signal[sr]

            #print("s:", s)
            #print("n:", n)
            #print("x:", x)

            # From object member variables
            b = tf.constant(self.SR_b[i],dtype=float)

            bsys = tf.constant(self.SR_b_sys[i],dtype=float)

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
            #print(B)
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
            if np.sum(mf & ~(fix1 | fix2) > 0):
                for r1i, r2i in zip(r1[mf],r2[mf]):
                    print("r1: {0}, r2: {1}, s+b: {2}".format(r1i,r2i,s+b))
                raise ValueError("Found both solutions forbidden (and unfixable) for some MLEs! I'm pretty sure this shouldn't happen so I'm calling it an error/bug in the calculation") 
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
            seeds[sr]['theta'] = theta_MLE / bsys # Scaled by bsys to try and normalise variables in the fit. Input variables are scaled the same way.
        #print("seeds:", seeds)
        #quit()
        return seeds


def collider_analyses_from_long_YAML(yamlfile,replace_SR_names=False):
    """Read long format YAML file of analysis data into ColliderAnalysis objects"""
    #print("Loading YAML")
    d = yaml.safe_load(yamlfile)
    #print("YAML loaded")
    analyses = []
    SR_name_translation = {}
    nextID=0
    for k,v in d.items():
        if replace_SR_names:
            # TensorFlow cannot handle some characters, so switch SR names to something simple
            SR_name_translation = {"SR{0}".format(nextID+i): sr[0] for i,sr in enumerate(v["Signal regions"])}
            srs = [["SR{0}".format(i)]+sr[1:] for i,sr in enumerate(v["Signal regions"])]
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
        analyses += [ColliderAnalysis(k,srs,cov,cov_order,ucz)]
    if replace_SR_names:
        return analyses, SR_name_translation
    else:
        return analyses
