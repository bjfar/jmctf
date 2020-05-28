"""Classes and functions associated with creating and fitting joint distribution objects"""

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import massminimize as mm
from . import common as c

def neg2LogL(pars,const_pars,analyses,data,pre_scaled_pars,transform=None):
    """General -2logL function to optimise"""
    if transform is not None:
        pars_t = transform(pars)
    else:
        pars_t = pars
    if const_pars is None:
        all_pars = pars_t
    else:
        all_pars = com.deep_merge(const_pars,pars_t)
    #print("all_pars:",all_pars)
    joint = JointDistribution(analyses.values(),all_pars,pre_scaled_pars)
    q = -2*joint.log_prob(data)
    total_loss = tf.math.reduce_sum(q)
    return total_loss, q, None, None

def optimize(pars,const_pars,analyses,data,pre_scaled_pars,transform=None,log_tag='',verbose=False):
    """Wrapper for optimizer step that skips it if the initial guesses are known
       to be exact MLEs"""

    opts = {"optimizer": "Adam",
            "step": 0.05,
            "tol": 0.01,
            "grad_tol": 1e-4,
            "max_it": 100,
            "max_same": 5,
            "log_tag": log_tag,
            "verbose": verbose 
            }

    kwargs = {'const_pars': const_pars,
              'analyses': analyses,
              'data': data,
              'pre_scaled_pars': pre_scaled_pars,
              'transform': transform
              }

    exact_MLEs = True
    for a in analyses.values():
        if not a.exact_MLEs: exact_MLEs = False # TODO: check implementations 
    if exact_MLEs:
        if verbose: print("All starting MLE guesses are exact: skipping optimisation") 
        total_loss, q, none, none = neg2LogL(pars,**kwargs)
    else:
        if verbose: print("Beginning optimisation")
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
        all_pars = com.deep_merge(const_pars,pars_t)
    joint = JointDistribution(analyses.values(),all_pars,pre_scaled_pars)
    return joint, q

class JointDistribution(tfd.JointDistributionNamed):
    """Object to combine analyses together and treat them as a single
       joint distribution. Uses JointDistributionNamed for most of the
       underlying work.
       
       analyses - list of analysis-like objects to be combined
       signal - dictionary of dictionaries containing signal parameter
        or constant tensors for each analysis.
       model_type - Specify treame
       pre_scaled_pars - If true, all input parameters are already scaled such that MLEs
                         have variance of approx. 1 (for more stable fitting).
                         If false, parameters are conventionally (i.e. not) scaled, and
                         required scaling internally.
                       - 'all', 'nuis', None

       TODO: This object has a bunch of stuff that only works with BinnedAnalysis
             objects as the 'analyses'. Needs to be generalised. 
    """
   
    def __init__(self, analyses, pars=None, pre_scaled_pars=None):
        self.analyses = {a.name: a for a in analyses}
        self.Osamples = {}
        for a in self.analyses.values():
           self.Osamples.update(c.add_prefix(a.name,a.get_observed_samples()))
        if pars is not None:
            #print("pars:", pars)
            self.pars = self.scale_pars(pars,pre_scaled_pars)
            dists = {} 
            self.Asamples = {}
            for a in self.analyses.values():
                d = c.add_prefix(a.name,a.tensorflow_model(self.pars[a.name]))
                dists.update(d)
                self.Asamples.update(c.add_prefix(a.name,a.get_Asimov_samples(self.pars[a.name])))
            super().__init__(dists) # Doesn't like it if I use self.dists, maybe some construction order issue...
            self.dists = dists
        # If no pars provided can still fit the analyses, but obvious cannot sample or compute log_prob etc.
        # TODO: can we fail more gracefully if people try to do this?
        #       Or possibly the fitting stuff should be in a different object? It seems kind of nice here though.

    def biased_sample(self, N, bias=1):
       """Sample from biased versions of all analyses and return them along their with sampling probability.
          For use in importance sampling.
          'bias' parameter indicates how many 'sigma' of upward bias to apply to the sample generation, computed
          in terms of sqrt(variance) of the background.
          Bias only applied to 'signal' parameters, not nuisance parameters.
          NOTE: This doesn't really work super well. Importance sampling is a bit tricky, might need smarter
                way of choosing the 'importance' distribution.
       """
       biased_analyses = copy.deepcopy(self.analyses)
       for a in biased_analyses.values():
           a.SR_b = a.SR_b + bias*np.sqrt(a.SR_b)
       biased_joint = JointDistribution(biased_analyses.values(), self.pars, pre_scaled_pars='all')
       samples = biased_joint.sample(N)
       logw = self.log_prob(samples) - biased_joint.log_prob(samples) # log(weight) for each sample
       return samples, logw

    def scale_pars(self,pars,pre_scaled_pars):
        """Prepare default nuisance parameters and return scaled signal and nuisance parameters for each analysis
           (scaled such that MLE's in this parameterisation have
           variance of approx. 1"""
        scaled_pars = {}
        #print("pars:",pars)
        for a in self.analyses.values():
            if a.name not in pars.keys(): raise KeyError("Parameters for analysis {0} not found!".format(a.name)) 
            s_pars, s_nuis, us_nuis = a.scale_pars(pars[a.name],pre_scaled_pars)

            # Logic to avoid applying scaling to parameters supplied with scaling already applied
            if pre_scaled_pars is None:
                #print("Scaling input parameters...")
                scaled_pars[a.name] = {**s_pars, **s_nuis}    
            elif pre_scaled_pars=='nuis':
                #print("Scaling only signal parameters: nuisance parameters already scaled...")
                scaled_pars[a.name] = {**s_pars, **us_nuis}
            elif pre_scaled_pars=='all':
                #print("No scaling applied: all parameters already scaled...")
                scaled_pars[a.name] = pars[a.name]

        return scaled_pars 

    def descale_pars(self,pars):
        """Remove scaling from parameters. Assumes they have all been scaled and require de-scaling."""
        descaled_pars = {}
        for a in self.analyses.values():
          if a.name in pars.keys():
            descaled_pars[a.name] = a.descale_pars(pars[a.name])
        return descaled_pars

    def get_nuis_parameters(self,signal,samples):
        """Samples vector and signal provided to compute good starting guesses for parameters"""
        pars = {}
        for a in self.analyses.values():
            pars[a.name] = a.get_nuisance_tensorflow_variables(self.get_samples_for(a.name,samples),signal[a.name])
        return pars

    def get_samples_for(self,name,samples):
        """Extract the samples for a specific analysis from a sample dictionary, and
           remove the analysis name prefix from the keys"""
        d = {key:val for key,val in samples.items() if key.startswith("{0}::".format(name))}
        return c.remove_prefix(name,d)

    def get_all_parameters(self,samples):
        """Samples vector and signal provided to compute good starting guesses for parameters"""
        pars = {}
        for a in self.analyses.values():
            pars[a.name] = a.get_all_tensorflow_variables(self.get_samples_for(a.name,samples))
        return pars

    def get_sample_structure(self):
        """Returns a dictionary whose structure is the same as samples from the joint PDF"""
        out = {}
        for a in self.analyses.values():
            out.update(c.add_prefix(a.name,a.get_sample_structure()))
        return out

    def fit_nuisance(self,signal,samples,log_tag='',verbose=False):
        """Fit nuisance parameters to samples for a fixed signal
           (ignores parameters that were used to construct this object)"""
        nuis_pars = self.get_nuis_parameters(signal,samples)
        joint_fitted, q = optimize(nuis_pars,signal,self.analyses,samples,pre_scaled_pars='nuis',log_tag=log_tag,verbose=verbose)
        return q, joint_fitted, nuis_pars

    def fit_nuisance_and_scale(self,signal,samples,log_tag='',verbose=False):
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
            out = com.deep_merge(nuis_pars,sig_out) # Return all signal and nuisance parameters, but not mu.
            return out 
        pars['mu'] = muV # TODO: Not attached to any analysis, but might work anyway  
        joint_fitted, q = optimize(pars,None,self.analyses,samples,pre_scaled_pars='nuis',transform=mu_to_sig,log_tag=log_tag,verbose=verbose)
        return q, joint_fitted, pars
  
    def fit_all(self,samples,fixed_pars={},log_tag='',verbose=False):
        """Fit all signal and nuisance parameters to samples
           (ignores parameters that were used to construct this object)
           Some special parameters within analyses are also flagged as
           un-fittable, e.g. theory uncertainty parameters. If these
           aren't provided then analyses will use default fixed values,
           but they can be supplied via the "fixed_pars" dict. These
           parameters are always treated as fixed, when it comes to
           starting MLE guesses etc.
        """
        # Make sure the samples are TensorFlow objects of the right type:
        samples = {k: tf.constant(x,dtype="float32") for k,x in samples.items()}
        all_pars = self.get_all_parameters(samples)
        # Deal with extra fixed parameters
        for analysis,pardict in fixed_pars.items():
            for par,val in pardict.items():
                if par not in all_pars[analysis].keys(): 
                    raise ValueError("Fixed parameter {0} for analysis {1} was not found among the parameters for that analysis! Please check that you have used the correct parameter name.".format(par,analysis)) 
                all_pars[analysis][par] = tf.constant(val, dtype=float, name=par)
        joint_fitted, q = optimize(all_pars,None,self.analyses,samples,pre_scaled_pars='all',log_tag=log_tag,verbose=verbose)
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
            joint = JointDistribution(self.analyses.values(),inpars,pre_scaled_pars=None)
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
        #print("Computing Hessian and various matrix operations for all samples...")
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
        #print("...done!")
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
                    raise ValueError("No test signals provided for parameter {0} in analysis {1}".format(kp,ka))
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
        joint = JointDistribution(self.analyses.values(),com.deep_merge(signal,theta_prof_dict),pre_scaled_pars=None)
        q = -2*joint.log_prob(samples)
        return q

