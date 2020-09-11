"""Analysis class for 'binned' analyses, whose joint PDF can be described
   as the product of independent Poission distributions for each bin,
   with multinormal contributions (e.g. associated with measurement of background rates)

   i.e.

   f(X,Y) = \prod_{i=1}^{N}[ Poission(X|s_i + b_i + theta_i) ] * MultiNormal(Y|theta)

   (theta_i and Y typically defined relative to given b_i determined from a real 
    background/control measurement, such that Y=0 for the 'observed' control measurement,
    by definition, and theta_i=0 would be the MLE for theta_i based only on that control
    measurement. However Y can take on other values during MC simulation) 
"""

import numpy as np
import tensorflow as tf
import copy
from tensorflow_probability import distributions as tfd
from .base_analysis import BaseAnalysis
from . import common as c

# Want to convert all this to YAML. Write a simple container to help with this.
class BinnedAnalysis(BaseAnalysis):
    def __init__(self,name,srs=None,cov=None,cov_order=None,unlisted_corr_zero=False,verify=True):
        super().__init__(name)
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
            self.exact_MLEs = False
        else:
            # Let driver classes know that we can analytically provide exact MLEs, so no numerical fitting is needed.
            self.exact_MLEs = False

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
            cov = tf.constant(self.cov,dtype=c.TFdtype)
            cov_diag = tf.constant([self.cov[k][k] for k in range(len(self.cov))])
            # Select which systematic to use, depending on whether SR participates in the covariance matrix
            bsys_tmp = [np.sqrt(self.cov_diag[cov_order.index(sr)]) if self.in_cov[i] else self.SR_b_sys[i] for i,sr in enumerate(self.SR_names)]
        else:
            bsys_tmp = self.SR_b_sys[:]

        # Prepare input parameters
        #print("input pars:",pars)
        s = pars['s'] * self.s_scaling # We "scan" normalised versions of s, to help optimizer
        theta = pars['theta'] * self.theta_scaling # We "scan" normalised versions of theta, to help optimizer
        #print("de-scaled pars: s    :",s)
        #print("de-scaled pars: theta:",theta)
        theta_safe = theta
  
        #print("theta_safe:", theta_safe)
        #print("rate:", s+b+theta_safe)
 
        # Expand dims of internal parameters to match input pars. Right-most dimension is the 'event' dimension, 
        # i.e. parameters for each independent Poisson distribution. The rest go into batch_shape.
        if s.shape==():
            n_batch_dims = 0
        else:
            n_batch_dims = len(s.shape) - 1
        new_dims = [1 for i in range(n_batch_dims)]
        b = tf.constant(self.SR_b,dtype=c.TFdtype)
        bsys = tf.constant(bsys_tmp,dtype=c.TFdtype)
        if n_batch_dims>0:
            b = tf.reshape(b,new_dims+list(b.shape))
            bsys = tf.reshape(bsys,new_dims+list(bsys.shape))
 
        # Poisson model
        poises0  = tfd.Poisson(rate = tf.abs(s+b+theta_safe)+c.reallysmall) # Abs works to constrain rate to be positive. Might be confusing to interpret BF parameters though.
        # Treat SR batch dims as event dims
        poises0i = tfd.Independent(distribution=poises0, reinterpreted_batch_ndims=1)
        tfds["n"] = poises0i

        # Multivariate background constraints
        if self.cov is not None:
            #print("theta_safe:",theta_safe)
            #print("covi:",self.covi)
            theta_cov = tf.gather(theta_safe,self.covi,axis=-1)
            #print("theta_cov:",theta_cov)
            cov_nuis = tfd.MultivariateNormalFullCovariance(loc=theta_cov,covariance_matrix=cov)
            tfds["x_cov"] = cov_nuis
            #print("str(cov_nuis):", str(cov_nuis))

            # Remaining uncorrelated background constraints
            if np.sum(~self.in_cov)>0:
                nuis0 = tfd.Normal(loc = theta_safe[...,~self.in_cov], scale = bsys[...,~self.in_cov])
                # Treat SR batch dims as event dims
                nuis0i = tfd.Independent(distribution=nuis0, reinterpreted_batch_ndims=1)
                tfds["x_nocov"] = nuis0i
        else:
            # Only have uncorrelated background constraints
            nuis0 = tfd.Normal(loc = theta_safe, scale = bsys)
            # Treat SR batch dims as event dims
            nuis0i = tfd.Independent(distribution=nuis0, reinterpreted_batch_ndims=1)
            tfds["x"] = nuis0i 
        #print("hello3")

        return tfds #, sample_layout, sample_count

    def add_default_nuisance(self,pars):
        """Prepare parameters to be fed to tensorflow model

           Provides default ("nominal") values for nuisance
           parameters if they are not specified.

           Input is full parameter dictionary, with SCALED
           parameters
        """
        pars_out = {}
        if 'theta' not in pars.keys():
            # If values not provided, trigger shortcut to set nuisance parameters to zero. Useful for sample generation.
            theta = tf.constant(0*pars['s'])
        else:
            theta = pars['theta']
        pars_out['s'] = pars['s']
        pars_out['theta'] = theta
        return pars_out

    def scale_pars(self,pars):
        """Apply scaling (to adjust MLEs to have var~1) to any valid 
        parameters found in input pars dictionary"""
        scaled_pars = {}
        if 's' in pars.keys():
            scaled_pars['s'] = pars['s'] / self.s_scaling
        if 'theta' in pars.keys(): 
            scaled_pars['theta'] = pars['theta'] / self.theta_scaling
        return scaled_pars

    def descale_pars(self,pars):
        """Remove scaling from parameters. Assumes they have all been scaled and require de-scaling."""
        descaled_pars = {}
        if 's' in pars.keys():
            descaled_pars['s'] = pars['s'] * self.s_scaling
        if 'theta' in pars.keys(): 
            descaled_pars['theta'] = pars['theta'] * self.theta_scaling
        return descaled_pars

    def get_Asimov_samples(self,signal_pars):
        """Construct 'Asimov' samples for this analysis
           Used to detemine asymptotic distributions of 
           certain test statistics.

           Assumes target MLE value for nuisance 
           parameters is zero.

           Requires unit-scaled parameters as input
        """
        Asamples = {}
        s = signal_pars['s'] * self.s_scaling
        b = tf.expand_dims(tf.constant(self.SR_b,dtype=c.TFdtype),0) # Expand to match shape of signal list 
        #print("Asimov s:",s)
        #print("self.in_cov:", self.in_cov)
        Asamples["n"] = tf.expand_dims(b + s,0) # Expand to sample dimension size 1
        if self.cov is not None:
            Asamples["x_cov"] = tf.constant(np.zeros((1,s.shape[0],np.sum(self.in_cov))),dtype=c.TFdtype)
            if np.sum(~self.in_cov)>0:
                Asamples["x_nocov"] = tf.constant(np.zeros((1,s.shape[0],np.sum(~self.in_cov))),dtype=c.TFdtype)
        else:
            Asamples["x"] = tf.expand_dims(0*s,0)
        #print("{0}: Asamples: {1}".format(self.name, Asamples))
        return Asamples

    def get_observed_samples(self):
        """Construct dictionary of observed data for this analysis
           Shapes should match the event_shapes of the tensorflow model
           for this analysis"""
        Osamples = {}
        Osamples["n"] = tf.constant(self.SR_n,dtype=c.TFdtype)
        if self.cov is not None:
            Osamples["x_cov"] = tf.constant([0]*np.sum(self.in_cov),dtype=c.TFdtype)
            if np.sum(~self.in_cov)>0:
                Osamples["x_nocov"] = tf.constant([0]*np.sum(~self.in_cov),dtype=c.TFdtype)
        else:
            Osamples["x"] = tf.constant([0]*len(self.SR_names),dtype=c.TFdtype)
        return Osamples

    def interest_parameter_shapes(self):
        """Get a dictionary describing the structure of the "interesting" parameters in this analysis
           Basically just the keys of the parameter dictionaries plus primitive (non-batch dimensions) 
           shape of each entry"""
        return {"s": (len(self.SR_n),)}

    def fixed_parameter_shapes(self):
        """Get a dictionary describing the structure of the fixed parameters in this analysis
           (i.e. parameters that can be altered along with the signal hypothesis in nuisance
           parameter fits, but which are kept fixed during all fitting.
           Basically just the keys of the parameter dictionaries plus primitive (non-batch dimensions) 
           shape of each entry"""
        return {}

    def nuisance_parameter_shapes(self):
        """Get a dictionary describing the nuisance parameter structure of this analysis.
           Basically just the keys of the parameter dictionaries plus primitive (non-batch dimensions) 
           shape of each entry"""
        return {"theta": (len(self.SR_b),)} # Just one nuisance parameter per signal region, packaged into one 1D tensor. 

    def get_nuisance_parameters(self,sample_dict,fixed_pars):
        """Get nuisance parameters to be optimized, for input to "tensorflow_model"""
        seeds = self.get_seeds_nuis(sample_dict,fixed_pars) # Get initial guesses for nuisance parameter MLEs
        stacked_seeds = np.stack([seeds[sr]['theta'] for sr in self.SR_names],axis=-1)
        free_pars = {"theta": stacked_seeds} 
        fixed_pars_out = {"s": fixed_pars["s"]} 
        return free_pars, fixed_pars

    def get_all_parameters(self,sample_dict,fixed_pars):
        """Get all parameters (signal and nuisance) to be optimized, for input to "tensorflow_model"""
        seeds = self.get_seeds_s_and_nuis(sample_dict) # Get initial guesses for parameter MLEs
        stacked_theta = np.stack([seeds[sr]['theta'] for sr in self.SR_names],axis=-1)
        stacked_s     = np.stack([seeds[sr]['s'] for sr in self.SR_names],axis=-1)
        free_pars = {"s": stacked_s, "theta": stacked_theta}
        fixed_pars = {}
        return free_pars, fixed_pars
        
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

        n_all = samples["n"]
        if self.cov is not None:
            xcov_all = samples["x_cov"]
            if np.sum(~self.in_cov)>0:
                xnocov_all = samples["x_nocov"]
        else:
            x_all = samples["x"]

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
            b = tf.constant(self.SR_b[i],dtype=c.TFdtype)
            bsys = tf.constant(self.SR_b_sys[i],dtype=c.TFdtype)

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

            seeds[sr]['theta'] = theta_MLE #/ self.theta_scaling[i]
            seeds[sr]['s']     = s_MLE #/ self.s_scaling[i]

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

        n_all = samples["n"]
        if self.cov is not None:
            xcov_all = samples["x_cov"]
            if np.sum(~self.in_cov)>0:
                xnocov_all = samples["x_nocov"]
        else:
            x_all = samples["x"]
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
            b = tf.constant(self.SR_b[i],dtype=c.TFdtype)
            bsys = tf.constant(self.SR_b_sys[i],dtype=c.TFdtype)

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
            seeds[sr]['theta'] = theta_MLE #/ self.theta_scaling[i] # Scaled by bsys to try and normalise variables in the fit. Input variables are scaled the same way.
        #print("seeds:", seeds)
        return seeds
