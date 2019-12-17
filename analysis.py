import numpy as np
import yaml
import pandas as pd
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import massminimize as mm
import copy

# Stuff to help format YAML output
class blockseqtrue( list ): pass
def blockseqtrue_rep(dumper, data):
        return dumper.represent_sequence( u'tag:yaml.org,2002:seq', data, flow_style=True )
yaml.add_representer(blockseqtrue, blockseqtrue_rep)

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
        #print("l_safe:", s+b+theta_safe)
 
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

qtrace = []
def neg2LogL(pars,const_pars,analyses,data,pre_scaled_pars):
    """General -2logL function to optimise"""
    if const_pars is None:
        all_pars = pars
    else:
        all_pars = deep_merge(const_pars,pars)
    #print("all_pars:",all_pars)
    joint = JMCJoint(analyses,all_pars,pre_scaled_pars)
    q = -2*joint.log_prob(data)
    global qtrace
    qtrace += [q]
    total_loss = tf.math.reduce_sum(q)
    return total_loss, q, None, None

def optimize(pars,const_pars,analyses,data,pre_scaled_pars,log_tag=''):
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
              'pre_scaled_pars': pre_scaled_pars
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
    if const_pars is None:
        all_pars = pars
    else:
        all_pars = deep_merge(const_pars,pars)
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

    def fit_all(self,samples,log_tag=''):
        """Fit all signal and nuisance parameters to samples
           (ignores parameters that were used to construct this object)"""
        all_pars = self.get_all_parameters(samples)
        joint_fitted, q = optimize(all_pars,None,self.analyses,samples,pre_scaled_pars='all',log_tag=log_tag)
        return q, joint_fitted, all_pars

    def cat_pars(self,pars):
        """Stack tensorflow parameters in known order"""
        parlist = []
        for ka,a in pars.items():
            for kp,p in a.items():
                parlist += [p]
        return tf.Variable(tf.concat(parlist,axis=-1),name="all_parameters")               

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

    def Hessian(self,samples):
        """Obtain Hessian matrix (and grad) at input parameter point"""

        # Stack current parameter values to single tensorflow variable for
        # easier matrix manipulation
        catted_pars = self.cat_pars(self.pars)
        #print("catted_pats:", catted_pars)
        with tf.GradientTape(persistent=True) as tape:
            pars = self.uncat_pars(catted_pars) # need to unstack for use in each analysis
            joint = JMCJoint(self.analyses,pars,pre_scaled_pars='all')
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
        Ni = 0 # Track how many interest/nuisance parameters there are in total
        Nn = 0
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
                    Ni += N
                else:
                    interest_p[ka][kp] = p
                    interest_i[ka][kp] = (i, N)
                    Nn += N
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
        Hii = self.sub_Hessian(H,parsi,parsj)
        Hjj = self.sub_Hessian(H,parsj,parsj) # Actually we don't need this block for now.
        Hij = self.sub_Hessian(H,parsi,parsj) #Off-diagonal block. Symmetric so we don't need both.
        return Hii, Hjj, Hij

    def quad_loglike_prep(self,samples):
        """Compute second-order Taylor expansion of log-likelihood surface
           around input parameter point(s), and compute quantities needed
           for analytic determination of profile likelihood for fixed signal
           parameters, under this approximation."""
        print("Computing Hessian and various matrix operations for all samples...") 
        H, g = self.Hessian(samples)
        interest_i, interest_p, nuisance_i, nuisance_p = self.decomposed_parameters(self.pars)
        Hii, Hnn, Hin = self.decompose_Hessian(H,interest_i,nuisance_i)
        Hnn_inv = tf.linalg.inv(Hnn)
        gn = self.sub_grad(g,nuisance_i)
        A = tf.linalg.matvec(Hnn_inv,gn)
        B = tf.linalg.matmul(Hnn_inv,Hin)
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
        f = mm.tools.func_partial(self.neg2loglike_quad,samples=samples,A=A,B=B,interest=interest_p,nuisance=nuisance_p)
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
        theta_0 = self.cat_pars(nuisance) # stacked nuisance parameter values at expansion point
        #print("theta_0.shape:",theta_0.shape)
        #print("A.shape:",A.shape)
        #print("B.shape:",B.shape)
        #print("s.shape:",s.shape)
        theta_prof = theta_0 - tf.expand_dims(A,axis=1) - tf.linalg.matvec(tf.expand_dims(B,axis=1),tf.expand_dims(s,axis=0))
        # de-stack theta_prof
        theta_prof_dict = self.uncat_pars(theta_prof,pars_template=nuisance)
        # Compute -2*log_prop
        joint = JMCJoint(self.analyses,deep_merge(signal,theta_prof_dict),pre_scaled_pars=None)
        q = -2*joint.log_prob(samples)
        return q

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
