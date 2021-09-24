"""Classes related to performing look-elsewhere-effect (LEE) corrections"""

import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import pandas as pd
import sqlite3
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import copy
import pathlib
import time
import io
import codecs
import h5py
from progress.bar import Bar
from progress.spinner import Spinner
from . import sql_helpers as sql
from . import common as c
from .binned_analysis import BinnedAnalysis
from .joint import JointDistribution

def LEEcorrection(analyses,signal,nosignal,name,N,fitall=True):
    """Compute LEE-corrected p-value to exclude the no-signal hypothesis, with
       'signal' providing the set of signals to consider for the correction
       
       Also computes local p-values for all input signal hypotheses.

       NOTE: This routine is somewhat old and deprecated. Use LEECorrectionMaster
       class below to do this in a smarter way, where samples and fit results are
       saved/loaded from database files and can be updated with more samples, for
       much better operation with large simulations.
    """
    print("Simulating {0}".format(name))
    
    # Create joint distributions
    joint   = JointDistribution(analyses) # Version for fitting (object is left with fitted parameters upon fitting)
    joint0  = JointDistribution(analyses,nosignal)
    joint0s = JointDistribution(analyses,signal)
    
    # Get Asimov samples for nuisance MLEs with fixed signal hypotheses
    samplesAb = joint0.Asamples
    samplesAsb = joint0s.Asamples 
 
    # Get observed data
    obs_data = joint0.Osamples
    
    # Evaluate distributions for Asimov datasets, in case where we
    # know that the MLEs for those samples are the true parameters
    qsbAsb = -2*(joint0s.log_prob(samplesAsb))
    qbAb   = -2*(joint0.log_prob(samplesAb))
    
    # Fit distributions for Asimov datasets for the other half of each
    # likelihood ratio
    print("Fitting w.r.t Asimov samples")
    log_prob_bAsb, joint_fitted, pars = joint.fit_nuisance(samplesAsb, nosignal, log_tag='bAsb')
    log_prob_sbAb, joint_fitted, pars = joint.fit_nuisance(samplesAb, signal, log_tag='sbAsb')
    qbAsb = -2*log_prob_bAsb
    qsbAb = -2*log_prob_sbAb
  
    qAsb = (qsbAsb - qbAsb)[0] # extract single sample result
    qAb = (qsbAb - qbAb)[0]
    
    # Generate background-only pseudodata to be fitted
    samples0 = joint0.sample(N)
    onesample0 = joint0.sample(1) # For some quick stuff
     
    # Generate signal pseudodata to be fitted
    samples0s = joint0s.sample(N)
   
    if fitall:
        print("Fitting GOF w.r.t background-only samples")
        log_prob_gof_b, joint_gof_fitted_b, gof_pars_b  = joint.fit_all(samples0, log_tag='gof_all_b')
        qgof_b = -2*log_prob_gof_b
        #print("Fitting GOF w.r.t signal samples")
        #log_prob_gof_sb, joint_gof_fitted_sb ,gof_pars_sb  = joint.fit_all(samples0s, log_tag='gof_all_sb')
        qgof_sb = -2*log_prob_gof_sb

        print("Fitting w.r.t background-only samples")
        log_prob_b , joint_fitted_b, nuis_pars_b = joint.fit_nuisance(samples0, nosignal, log_tag='qb')
        log_prob_sb, joint_fitted_sb, nuis_pars_s = joint.fit_nuisance(samples0, signal, log_tag='qsb')
        qb  = -2*log_prob_b
        qsb = -2*log_prob_sb
        q = qsb - qb # mu=0 distribution
    else:
        # Only need the no-signal nuisance parameter fits for quadratic approximations
        print("Fitting no-signal nuisance parameters w.r.t background-only samples")
        log_prob_b, joint_fitted_b, nuis_pars_b = joint.fit_nuisance(samples0, nosignal, log_tag='qb')
        qb = -2*log_prob_b

        # Do one full GOF fit just to determine parameter numbers 
        null, null, gof_pars_b  = joint.fit_all(onesample0)

    # Obtain function to compute neg2logl for fitted samples, for any fixed input signal,
    # with nuisance parameters analytically profiled out using a second order Taylor expansion
    # about the GOF best fit points.
    #print("fitted pars:", gof_pars_b)
    #f1 = joint_gof_fitted_b.log_prob_quad_f(samples0)

    # What if we expand about the no-signal point instead? Just to see...
    f2 = joint_fitted_b.log_prob_quad_f(samples0)
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

    qsb_quad = -2*f2(signal)
    #qsb_quad = -2*combf(signal)
    #qb_quad = -2*f2(nosignal) # Only one of these so can easily do it numerically, but might be more consistent to use same approx. for both.
    #print("qsb_quad:", qsb_quad)
    #q_quad = qsb_quad - qb_quad
    q_quad = qsb_quad - qb # Using quad approx only for signal half. Biased, but maybe better p-value behaviour.

    # print("Fitting w.r.t signal samples")
    # log_prob_b_s , joint_fitted, nuis_pars = joint.fit_nuisance(samples0s, nosignal)
    # log_prob_sb_s, joint_fitted, nuis_pars = joint.fit_nuisance(samples0s, signal)
    # qb_s  = -2*log_prob_b_s
    # qsb_s = -2*log_prob_sb_s 
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
    cdf = c.eCDF(q_quad)
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
        q_min     = c.gather_by_idx(q,q_min_i)
   
        # Local p-values at selected point
        plocal_BF          = c.gather_by_idx(plocal_all,q_min_i).numpy()    
        sigmalocal_BF      = c.gather_by_idx(sigmalocal_all,q_min_i).numpy()

    #plocal_all_quad = tf.stack(plocal_quad,axis=-1)
    #sigmalocal_all_quad = tf.stack(sigmalocal_quad,axis=-1)
    #sigmalocal_min_quad_i = tf.argmax(sigmalocal_all_quad,axis=-1)
    q_min_quad_i = tf.argmin(q_quad,axis=-1)
    qquad_min = c.gather_by_idx(q_quad,q_min_quad_i) 
    sigmalocal_B_quad  = c.gather_by_idx(sigmalocal_all_quad,q_min_quad_i).numpy()
    plocal_BF_quad     = c.gather_by_idx(plocal_all_quad,q_min_quad_i).numpy()
  
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
    log_prob_bO , joint_fitted, pars = joint.fit_nuisance(obs_data, nosignal)
    log_prob_sbO, joint_fitted, pars = joint.fit_nuisance(obs_data, signal)
    qbO  = -2*log_prob_bO
    qsbO = -2*log_prob_sbO 
    qO = (qsbO - qbO)[0] # extract single sample result

    print("Fitting GOF w.r.t observed data")
    log_prob_gof_obs, joint_fitted, pars  = joint.fit_all(obs_data, log_tag='gof_obs')
    qgof_obs = -2*log_prob_gof_obs
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

def fix_dims(array):
    """Helper function for ensuring that parameter-related arrays have the
       correct dimensions for operations in LEECorrector classes"""
    if array.shape==():
       array = tf.expand_dims(array, axis=0)
 
    if len(array.shape)==1:
       out = tf.expand_dims(array, axis=1)
    elif len(array.shape)==3:
       out = tf.squeeze(array, axis=1)
    else:
       out = array
    return out

def fix_dims_quad(log_prob_quad,batch_shape):
    """Helper function to adjust dimensions of output of 'quad' functions"""
    # if log_prob_quad.shape==():
    #     log_prob_quad = tf.expand_dims(log_prob_quad,axis=0) # Add in "hypothesis" dimension if it was missing due to only one hypothesis. 
    # if len(log_prob_quad.shape)==1:
    #     log_prob_quad = tf.expand_dims(log_prob_quad,axis=0) # Add in "events" dimension if it was missing due to only one event.
    # #if len(log_prob_quad.shape)>2:
    # #    # We need 2D output, but input is already larger than 2D.
    # #    # Probably this is due to the existence of some singleton dimensions that we can remove.
    # #    # Try this before throwing error.
    # #    pass

    # New method: squeeze all singletons, then figure out which dims may need expanding back to 2D
    # To do this we need to inspect the batch_shape associated with the JointDistribution from with the quad
    # function was generated, so that we know if these dims would get squeezed to nothing.
    singleton_batch = all([x == 1 for x in batch_shape])
    
    singletons = [i for i,d in enumerate(log_prob_quad.shape) if d==1]
    if len(singletons)>0:
        log_prob_squeezed = tf.squeeze(log_prob_quad,axis=singletons)
        print("squeezed {} singleton dims".format(len(singletons)))
    else:
        log_prob_squeezed = log_prob_quad

    if log_prob_squeezed.shape==():
        print("Squeezed shape is scalar, adding in singleton hypothesis and event dimensions")
        lpq2D = tf.expand_dims(tf.expand_dims(log_prob_squeezed,axis=0),axis=0) # Add in "hypothesis" and "events" dimensions
    elif len(log_prob_squeezed.shape)==1:
        if singleton_batch:
            print("Squeezed shape is dim 1, and batch_shape is singleton, so adding in singleton event dim")
            lpq2D = tf.expand_dims(log_prob_squeezed,axis=0) # Add in "events" dimension if it was missing due to only one event.
        else:
            print("Squeezed shape is dim 1, and batch_shape is NOT singleton, so adding in hypothesis dim")
            lpq2D = tf.expand_dims(log_prob_squeezed,axis=1) # Must be the "hypothesis" dimension that is missing
    elif len(log_prob_squeezed.shape)==2:
        print("Squeezed shape is 2D, no change required")
        # Already 2D
        lpq2D = log_prob_squeezed
        pass
    else:
        msg = "Final shape of log_prob_quad array was invalid! Shape was originally {0}, and it should be squeezed/expanded to 2D (dim[0]=events, dim[1]=hypotheses), however this operation was not possible (squeezed shape was {1})".format(log_prob_quad.shape,log_prob_squeezed.shape)
        raise ValueError(msg)
    return lpq2D

class LEECorrectorBase:
    """Base class for LEE correector classes, containing common utilities such as SQL
       database access"""

    def __init__(self,path,dbname):
        self.db = '{0}/{1}.db'.format(path,dbname) 
        pathlib.Path(path).mkdir(parents=True, exist_ok=True) # Ensure output path exists

    def connect_to_db(self):
        conn = sqlite3.connect(self.db,detect_types=sqlite3.PARSE_DECLTYPES) # Need the second argument to detect new numpy type 
        return conn

    def close_db(self,conn):
        conn.commit()
        conn.close()

    def _load_results(self,table,columns,keys=None,primary='EventID',from_observed=False):
        if from_observed==True: table += "_observed"
        conn = self.connect_to_db()
        cur = conn.cursor()
        results = sql.load(cur,table,columns,keys,primary)
        info = sql.table_info(cur,table)

        # Figure out types to use for the requested columns
        #info: [(0, 'EventID', 'integer', 0, None, 1), (1, 'neg2logL_null', 'real', 0, None, 0), (2, 'neg2logL_profiled', 'real', 0, None, 0), (3, 'neg2logL_profiled_quad', 'real', 0, None, 0)]
        all_dtypedict = {'integer': 'i8', 'real': 'f8'}
        col_dtypedict = {col: all_dtypedict[dtype] for row, col, dtype, a, b, c in info}        
        dtypes = [(col, col_dtypedict[col]) for col in columns]
        arr = np.array(results,dtype=dtypes) 
        if primary in columns:
            index = primary
        else:
            index = None
        data = pd.DataFrame.from_records(arr,index)
        return data

    def load_results(self,table,columns,keys=None,primary='EventID',get_observed=False,from_observed=False):
        """Load rows from specified table into panda dataframe. 
           if get_observed==True we also grab the data from the matching 'observed' table.
           if from_observed==True we *only* grab the data from the matching 'observed' table.
        """
        sim = self._load_results(table,columns,keys,primary,from_observed)
        if get_observed==True:
           # Remove some columns that make no sense for the observed data
           rem = ["logw"]
           for r in rem:
               if r in columns: columns.remove(r)
           obs = self._load_results(table,columns,keys,primary,from_observed=True)
           return sim, obs
        else:
           return sim

class LEECorrectorMaster(LEECorrectorBase):
    """A class to wrap up analysis and database access routines to manage LEE corrections
       for large numbers of analyses, with many alternate hypotheses to test, and with many
       random MC draws. Allows more alternate hypotheses and random draws to be added to the
       simulation without recomputing everything"""

    def __init__(self,analyses,path,master_name,null_hyp,nullname):
        super().__init__(path,"{0}_{1}".format(master_name,nullname))

        self.LEEanalyses = {}
        self.profiled_table = 'profiled' # For combined results profiled over all alternate hypotheses
        self.nullname = nullname # For combined results for null hypothesis
        self.local_table = 'local_' # Base name for tables related to specific alternate hypotheses whose local properties we want to investigate
        self.null_table = self.local_table+self.nullname # Name of table contained null hypothesis fit results (and related)
        self.analyses = analyses # For introspection only. Otherwise LEECorrectorAnalysis interface is used
        for a in self.analyses:
            self.LEEanalyses[a.name] = LEECorrectorAnalysis(a,path,master_name,{a.name: null_hyp[a.name]},nullname)

        conn = self.connect_to_db()
        cur = conn.cursor()

        # Table for recording final profiled (i.e. best-fit) test statistic results
        comb_cols = [ ("EventID", "integer primary key")
                     ,("log_prob", "real")
                     ,("log_prob_quad", "real")
                     ,("logw", "real")]
        colnames = [x[0] for x in comb_cols]

        sql.create_table(cur,self.profiled_table,comb_cols)
        sql.create_table(cur,self.profiled_table+"_observed",comb_cols[:-1]) # Don't want the logw column for this
        self.close_db(conn)
 
    def parameter_shapes(self):
        """Introspect the parameters for the joint distribution of all analyses.
           For helping understand what null hypotheses parameters are required as
           input.
        """
        # JointDistribution already has a method for doing this
        return JointDistribution(self.analyses).parameter_shapes()

    def decomposed_parameter_shapes(self):
        """Introspect the parameters for the joint distribution of all analyses,
           decomposed into nuisance, interest, and fixed parameters.
           For helping understand what null hypotheses parameters are required as
           input.
        """
        # JointDistribution already has a method for doing this
        return JointDistribution(self.analyses).decomposed_parameter_shapes()

    def add_events(self,N,bias=0):
        """Generate pseudodata for all analyses under the null hypothesis"""
        logw = tf.zeros((int(N),1))
        for name,a in self.LEEanalyses.items():
            logw += a.add_events(N,bias)
        # Record the eventIDs in the combined database, plus the combined weights
        self.add_events_comb(N,logw)

    def add_events_comb(self,N,logw):
        """Record existence of events in the profiled table"""
        # Do this by just inserting null data and letting the EventID column auto-increment
        conn = self.connect_to_db()
        cur = conn.cursor()
        command = "INSERT INTO {0} (log_prob,logw) VALUES (?,?)".format(self.profiled_table)
        vals = [(None,logwi[0]) for logwi in logw.numpy().tolist()]
        cur.executemany(command,  vals) 
        self.close_db(conn)

    def process_null(self):
        """Perform nuisance parameter fits of null hypothesis for all events where it hasn't already been done"""
        self.process_alternate_local() # Special case of local alternate hypothesis processing for the null hypothesis case.

    def process_alternate_local(self,alt_hyp=None,name=None,do_quad=True,event_batch_size=10000):
        """Perform nuisance parameter fits of single null/alternate hypotheses for all events where it hasn't already been done"""
        for a in self.LEEanalyses.values():
            if alt_hyp is None:
                a.process_null()
            elif name is None:
                raise ValueError("Name for alternate hypothesis needs to be provided, for identifying it in the results database!")
            else:
                a.process_alternate_local(alt_hyp,name)
 
        # Fits completed; extract all results and get combined log_prob values
        comb = None
        for a in self.LEEanalyses.values():
            if alt_hyp is None: table = a.local_table+a.nullname
            else: table = a.local_table+name
            df = a.load_results(table,['EventID','log_prob','log_prob_quad'])
            #print("df:",df)
            if comb is None: 
                comb = df
            else:
                comb += df
        conn = self.connect_to_db()
        cur = conn.cursor()
        if alt_hyp is None: table = self.null_table
        else: table = self.local_table+name
        sql.create_table(cur,table,[('EventID','integer primary key'),('log_prob','real'),('log_prob_quad','real')])
        sql.upsert(cur,table,comb,'EventID')
        self.close_db(conn)

        # Do the same for asymptotic/Asimov and observed values (non-null case only)
        if alt_hyp is not None:
            comb = None
            cols = ['qAsb','qAb','qO','log_prob']
            cols_quad = ['qAsb_quad','qAb_quad','qO_quad','log_prob_quad']
            all_cols = cols+cols_quad
            for a in self.LEEanalyses.values():
                table = a.local_table+name+"_observed"
                df = a.load_results(table,all_cols)
                if comb is None: 
                    comb = df
                else:
                    comb += df
            comb.index.rename('EventID',inplace=True)
            conn = self.connect_to_db()
            cur = conn.cursor()
            col_info = [('EventID','integer primary key')]+[(col,'real') for col in all_cols]
            sql.create_table(cur,self.local_table+name+'_observed',col_info)
            sql.upsert(cur,self.local_table+name+'_observed',comb,primary='EventID')
            self.close_db(conn)
        else:
            # Else just add the observed value for the null hypothesis likelihood
            comb = None
            cols = ['log_prob'] 
            cols_quad = ['log_prob_quad'] 
            all_cols = cols+cols_quad
            for a in self.LEEanalyses.values():
                table = a.local_table+a.nullname+"_observed"
                df = a.load_results(table,all_cols)
                if comb is None: 
                    comb = df
                else:
                    comb += df
            comb.index.rename('EventID',inplace=True)
            conn = self.connect_to_db()
            cur = conn.cursor()
            col_info = [('EventID','integer primary key')]+[(col,'real') for col in all_cols]
            sql.create_table(cur,self.null_table+'_observed',col_info)
            sql.upsert(cur,self.null_table+'_observed',comb,primary='EventID')
            self.close_db(conn)

    def _get_quads(self, EventIDs, **kwargs):
        """Get quadratic approximations to likelihood surfaces for the given EventIDs
           (using the null hypothesis (with fitted nuisance parameters) as the expansion point
           by default. Supplying other points is untested...)"""
        quads = {name: a._get_quad(EventIDs, **kwargs) for name,a in self.LEEanalyses.items()}
        return quads

    def _process_alternate_batch(self, alt_hyp_gen, EventIDs, dbtype):
        """For internal use in 'process_alternate' function. Processes a single batch of events."""
        quads = self._get_quads(EventIDs)

        if hasattr(alt_hyp_gen,'count') and hasattr(alt_hyp_gen,'chunk_size'):
            Ns = alt_hyp_gen.count
            Nchunk = alt_hyp_gen.chunk_size
            Nbatches = Ns // Nchunk
            rem = Ns % Nchunk
            if rem!=0: Nbatches+=1
            bar = Bar('Processing alternate hypotheses in batches of {0}'.format(Nchunk), max=Nbatches)
        else:
            bar = Spinner('Processing alternate hypotheses') # Unknown size

        max_log_probs = None
        loop_ran = False
        for r in alt_hyp_gen:
            loop_ran = True
            comb_log_probs = None

            # Check that user-supplied alt_hyp_gen function gave us
            # usable input. Since this is user-supplied we do particularly
            # careful checking
            # ---------------------------------
            gen_msg1 = "User supplied alternate hypothesis generator class ('alt_hyp_gen' argument) did not produce valid hypothesis data!"
            gen_msg2 = "The hypothesis dictionary should be of the structure {<analysis_name>: {<parameter_name>: <parameter_values>}}, where <parameter_values> is a 1 or 2D array of values for each parameters, with dim 0 being separate hypotheses, and dim 1 being entries of vector parameters (scalar parameters may omit this dimension).\nThe ID array should simply be a 1D array of ID numbers that uniquely identify each hypothesis given in the corresponding entries of the hypothesis dictionary."

            # Check that dict, ID tuple is returned
            try:
                alt_chunk, altIDs = r
            except ValueError as e:
                msg = gen_msg1 + " The 'next' method did not return a (dictionary,array) tuple (see rest of error for more information)\n" + gen_msg2
                raise ValueError(msg) from e
                
            # Check that dict part of tuple is a dict
            if not isinstance(alt_chunk, dict):
                msg = gen_msg1 + " The 'next' method did not return a (dictionary,array) tuple (the first member of the tuple was not a dictionary).\n"+gen_msg2
                raise ValueError(gen_msg)

            # Try to convert the dict bottom-level values to tensors
            alt_chunk = c.convert_to_TF_constants(alt_chunk)

            # Check that dict has correct structure
            gen_msg3 = "The dictionary returned by the 'next' method does not have the correct structure. It should be a dictionary of parameter dictionaries"
            hyp_size = None # Size of hypothesis dimension
            for a,pars in alt_chunk.items():
                if not isinstance(alt_chunk[a], dict):
                    msg = gen_msg1 + " " + gen_msg3 + ", but the values of the outer dict were not dicts." + gen_msg2
                    raise ValueError(msg)
                for p, vals in pars.items():
                    try:
                        vals.shape
                    except AttributeError as e:
                        msg = gen_msg1 + " " + gen_msg3 + ", however the parameter \"values\" given for parameter {0} in analysis {1} could not be interpreted as arrays (they did not have a 'shape' method).".format(p,a)
                        raise ValueError(msg) from e
                    if len(vals.shape)!=1 and len(vals.shape)!=2:
                        msg = gen_msg1 + " " + gen_msg3 + ", however the parameter values given are not the right shape! They should be 1/2D (see description below) but shape of {0} parameter in {1} analysis was {2}".format(p,a,vals.shape)
                        raise ValueError(msg)
                    if hyp_size is None: hyp_size = vals.shape[0]
                    elif hyp_size != vals.shape[0]:
                        msg = gen_msg1 + " " + gen_msg3 + ", however the parameter values given have inconsistent shapes! They should be 1/2D (see description below), but number of hypotheses returned in array for parameter {0} in analysis {1} was {2} (previous parameter arrays contained {3} hypotheses)".format(p,a,vals.shape[0],hyp_size)
                        raise ValueError(msg)
 
            # Check that ID array is valid
            try:
                len(altIDs)
            except TypeError:
                msg = gen_msg1 + ". The 'next' method did not return a (dictionary,array) tuple (the 'array' item had no 'len' property so cannot be interpreted as an array. It needs to be a 1D list/array giving unique ID numbers)"
                raise ValueError(msg)
            if len(altIDs) != hyp_size:
                msg = gen_msg1 + ". The array in the (dict,array) tuple returned by the 'next' method did not have a length consistent with the parameters given in 'dict'. From the dict it was inferred that there are {0} hypotheses in this chunk, however the ID array has length {1}".format(hyp_size,len(altIDs))
                raise ValueError(msg)
            # ---------------------------------

            # Input validated, onto the analysis

            for name, a in self.LEEanalyses.items():
                print("running quad:",name)
                quad = quads[name]
                pars = {name: alt_chunk[name]}
                print("EventIDs:", EventIDs)
                if isinstance(EventIDs, str) and EventIDs=="observed":
                    sample_shape = (1,)
                else:
                    sample_shape = (len(EventIDs),)
                print("pars:", pars)
                print("sample_shape:", sample_shape)
                batch_shape = a.joint.expected_batch_shape_nuis(pars, sample_shape=sample_shape)
                log_probs = quad(pars)
                lpq2D = fix_dims_quad(log_probs, batch_shape)
 
                #if len(log_probs.shape)==1:
                #    log_probs = tf.expand_dims(log_probs,axis=0) # Add in "events" dimension if it was missing due to only one event.
                #elif len(log_probs.shape)!=2:
                #    msg = "Shape of log_probs array returned from quadratic nuisance parameter estimating function ('compute_quad') for analysis {0} was invalid! Shape was {1}, but it should be 2D (dim[0]=events, dim[1]=hypotheses), or 1D if only one event is being processed.".format(name,log_probs.shape)
                #    raise ValueError(msg)

                print("name:", name)
                print("log_probs.shape:", log_probs.shape)
                print("batch_shape:", batch_shape)
                print("lpq2D.shape:", lpq2D.shape)

                # Record all alternate_hypothesis likelihoods to disk, so that we can use them for bootstrap resampling later on.
                # Warning: may take a lot of disk space if there are a lot of alternate hypotheses.
                #print("EventIDs:", EventIDs)
                a.record_alternate_logLs(lpq2D, altIDs, EventIDs, Ltype='quad', dbtype=dbtype)
                #print("...done")
                #print("alterate hypothesis log_probs:", lpq2D)
                if comb_log_probs is None:
                    comb_log_probs = lpq2D
                else:
                    comb_log_probs += lpq2D

            #print("alt_chunk:", alt_chunk)
            # Select the maximum logL from across all alternate hypotheses
            if max_log_probs is not None:
                all_log_probs = tf.concat([tf.expand_dims(max_log_probs,axis=-1),comb_log_probs],axis=-1)
            else:
                all_log_probs = comb_log_probs
            max_log_probs = tf.reduce_max(all_log_probs,axis=-1)
            bar.next()
        bar.finish()
        if not loop_ran:
            msg = "Problem processing alternate hypotheses! The user-supplied generator of hypothesis parameters did not yield any output!"
            raise ValueError(msg)
        elif max_log_probs is None:
            msg = "Problem processing alternate hypotheses! Result of batch was None!"
            raise ValueError(msg)
        return max_log_probs
 
    def process_alternate_observed(self,alt_hyp_gen,quad_only=True,dbtype='hdf5'):
        """Perform fits for all supplied alternate hypotheses, for just the *observed* data"""
        max_log_prob = self._process_alternate_batch(alt_hyp_gen,"observed",dbtype)
        
        # Extract data for null hypothesis (should be pre-computed by process_null)

              
        # Write the compute max_log_prob to disk for this batch of events
        data = pd.DataFrame(max_log_prob.numpy(),columns=['log_prob_quad'])
        data.index.name = 'EventID' 
 
        conn = self.connect_to_db()
        cur = conn.cursor()
        sql.upsert_if_larger(cur,self.profiled_table+"_observed",data,'EventID') # Only replace existing values if new ones are larger
        self.close_db(conn)


    def process_alternate(self,alt_hyp_gen,quad_only=True,new_events_only=False,event_batch_size=1000,dbtype='hdf5'):
        """Perform fits for all supplied alternate hypotheses, for all events in the database,
           and record the best-fit alternate hypothesis for each event. Compares to any existing best-fits
           in the database and updates if the new best-fit is better.
           If 'new_events_only' is true, fits are only performed for events in the database
           for which no alternate hypothesis fit results are yet recorded."""
        # First process the observed dataset
        print("Processing alternate hypotheses for *observed* dataset")
        self.process_alternate_observed(alt_hyp_gen(),quad_only,dbtype)
        print("...done!")
        Nevents = self.count_events_comb()
        still_processing = True
        offset = 0
        N_event_batches = int(np.ceil(Nevents / event_batch_size))
        batchi = 1; 
        while still_processing:
            print("Processing event batch {0} of {1} (batch_size={2})".format(batchi,N_event_batches,event_batch_size))
            if new_events_only:
                EventIDs = self.load_eventIDs(event_batch_size,self.profiled_table,'log_prob_quad is NULL')
            else:
                EventIDs = self.load_eventIDs(event_batch_size,offset=offset)
                offset += event_batch_size
            if EventIDs is None: still_processing = False
            if still_processing:
                max_log_prob = self._process_alternate_batch(alt_hyp_gen(),EventIDs,dbtype)              
                # Write the compute min_neg2logLs to disk for this batch of events
                data = pd.DataFrame(max_log_prob.numpy(),index=EventIDs.numpy(),columns=['log_prob_quad'])
                data.index.name = 'EventID' 
 
                conn = self.connect_to_db()
                cur = conn.cursor()
                sql.upsert_if_larger(cur,self.profiled_table,data,'EventID') # Only replace existing values if new ones are smaller
                self.close_db(conn)

                batchi+=1

    def get_bootstrap_sample(self,N,batch_size=2000,dbtype='hdf5'):
        """Obtain a bootstrap resampling of all 'full' tables in all analyses, 
           combine/profile the likelihoods, and add results to bootstrap table.

           Keep batch size small when number of alternate hypotheses is large, to avoid
           running out of RAM. Will have likelihoods for all bootstrap events
           for all alternate hypotheses."""
  
        all_max_log_prob = None
        all_b_log_prob = None
        if N=='all': # Special keyword to just re-do profiling rather than bootstrap resampling. To cross-check this profiling calculation with original calculation.
            print("Recomputing profile over alternate hypothesis for existing events (no resampling...)")
        else:
            Nbatches = int(np.ceil(N/batch_size))
            bar = Bar('Generating {0} bootstrap samples in batches of {1}'.format(N,batch_size), max=Nbatches*len(self.LEEanalyses))
        done = False
        i = 0
        while not done:
            if N!='all' and i>=Nbatches: done = True
            if not done:
                if N=='all':
                    size = (i*batch_size+1, (i+1)*batch_size) # will recompute profiling for events in this range (inclusive)
                else:
                    if i==Nbatches-1 and N % batch_size > 0: size = N % batch_size
                    else: size = batch_size
                all_log_probs = None
                these_b_log_probs = None
                for name,a in self.LEEanalyses.items():
                    s_log_probs, b_log_probs = a.get_bootstrap_sample(size,dbtype=dbtype)
                    if s_log_probs is None: 
                        done = True
                        break
                    #print("s_log_probs.shape:", s_log_probs.shape)
                    #print("b_log_probs.shape:", b_log_probs.shape)
                    if all_log_probs is None: all_log_probs = s_log_probs
                    else: all_log_probs += s_log_probs
                    if these_b_log_probs is None: these_b_log_probs = b_log_probs
                    else: these_b_log_probs += b_log_probs
                    if N=='all':
                        print("Analysis {0}, batch {1}".format(name, i))
                    else:
                        bar.next()


                if not done:
                    #print("all_log_probs.shape:",all_log_probs.shape)

                    # Profile
                    max_log_prob = tf.reduce_max(all_log_probs,axis=0) # Signal dimension is first here, different to elsewhere
                    #print("max_log_prob.shape:", max_log_prob.shape)
                            
                    # Write to disk? Could just return if this is fast. Test to find out.
                    if all_max_log_prob is None: all_max_log_prob = max_log_prob
                    else: all_max_log_prob = tf.concat([all_max_log_prob,max_log_prob],axis=0) # Only one dimension left
 
                    if all_b_log_prob is None: all_b_log_prob = these_b_log_probs
                    else: all_b_log_prob = tf.concat([all_b_log_prob,these_b_log_probs],axis=0)
            i+=1
        if N!='all': bar.finish()
        return all_max_log_prob, all_b_log_prob

    def load_eventIDs(self,N,reftable=None,condition=None,offset=0):
        """Loads eventIDs from database where 'condition' is true in 'reftable'
           To skip rows, set 'offset' to the first row to be considered."""
        conn = self.connect_to_db()
        cur = conn.cursor()

        if reftable is not None:
            # First see if any data is in the reference table yet:
            cur.execute("SELECT Count(EventID) FROM "+reftable)
            results = cur.fetchall()
            nrows = results[0][0]

        command = "SELECT A.EventID from {0} as A".format(self.profiled_table)        
        if reftable is not None and nrows!=0 and condition is not None:
           # Apply extra condition to get e.g. only events where "neg2LogL" column is NULL (or the EventID doesn't exist) in the 'background' table
           command += """
                      left outer join {0} as B
                          on A.EventID=B.EventID
                      where
                          B.{1} 
                      """.format(reftable,condition)
        command += " LIMIT {0} OFFSET {1}".format(N,offset)
        cur.execute(command)
        results = cur.fetchall()
        self.close_db(conn) 

        if len(results)>0:
            # Convert back into dictionary of tensorflow tensors
            # Start by converting to one big tensor
            EventIDs = tf.squeeze(tf.convert_to_tensor(results, dtype=tf.int32))
        else:
            EventIDs = None
        return EventIDs 

    def count_events_comb(self):
        """Count number of rows in the combined output table.
           Should be equal to max(EventID)+1 since we
           don't ever delete events."""
        conn = self.connect_to_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM {0}".format(self.profiled_table))
        results = cur.fetchall()
        self.close_db(conn) 
        return results[0][0]

    def ensure_equal_events(self):
        """Generate events in all analyses as needed for them to all have the same
           number of events. Should only be needed in case run was aborted mid-way or
           a file has been deleted or some such. Or if say a new analysis has been added."""
        raise RuntimeError("This function needs to be updated to deal with biased event generation!")
        N = []
        for name,a in self.LEEanalyses.items():
            N += [a.count_events()]

        # Now the combined table.
        combN = self.count_events_comb()
        maxN = max(N+[combN])

        for Ni,(name,a) in zip(N,self.LEEanalyses.items()):
            if Ni<maxN: a.add_events(maxN-Ni)   

        if combN<maxN: self.add_events_comb(maxN-combN)

    def add_alternate(self,alt_hyp):
        pass

    def add_to_database(self):
        pass

    def load_all_events(self,with_IDs=False):
        """Retrieve all events currently on disk. Mainly for manual inspection/debugging purposes.
           Events returned in dict format as if all analyses were part of one big JointDistribution object."""
        event_dict = {}
        EventIDs = None
        for aname, LEEa in self.LEEanalyses.items():
            EventIDs, events = LEEa.load_events()
            # EventIDs should be the same across analyses. TODO: could add a check for this
            event_dict.update(events)
        if with_IDs:
            return EventIDs, event_dict
        else:
            return event_dict

    def load_all_null_nuis_pars(self):
        """Retrieve all nuisance parameter fits to null hypothesis. Mainly for manual inspect/debugging purposes"""
        par_dict = {}
        EventIDs, event_dict = self.load_all_events(with_IDs=True)
        for aname, LEEa in self.LEEanalyses.items():
            pars = LEEa.load_null_nuis_pars(EventIDs)
            par_dict.update(pars)
        return par_dict

class LEECorrectorAnalysis(LEECorrectorBase):
    """A class to wrap up a SINGLE analysis with connection to output sql database,
       to be used as part of look-elsewhere correction calculations

       The structure of output is as follows:
        - 1 master directory for each complete analysis to be performed, containing:
          - 1 database file per Analysis class, containing:
            - 1 'master' table assigning IDs (just row numbers) to each 'event', or pseudoexperiment, and recording the data
            - 1 table synchronised with the 'master' table, containing local test statistic values associated with the (fixed) observed 'best fit' alternate hypothesis
              (may add more for other 'special' alternate hypotheses for which we want local test statistic stuff)
            - 1 table  "                            "                        test statistic values associated with the combined best fit point for each event
          - 1 datafiles file for the combination, containing:
            - 1 table containining combined test statistic values for each event (synchronised with each Analysis file)

        We have to compute test statistic values for *ALL* alternate hypotheses to be considered, for every event, however
        this is too much data to store. So we keep only the 'profiled' test statistic values, i.e. the values extremised
        over the available alternate hypotheses. If more alternate hypotheses are added, we can just compare to the already-computed
        extrema test statistic values for each event to see if they need to be updated.
    """

    def __init__(self,analysis,path,comb_name,null_hyp,nullname):
        super().__init__(path,analysis.name)

        self.event_table = 'events'
        self.nullname = nullname # String to identify null hypothesis tables
        self.local_table = 'local_' # Base name for tables containing local TS data)
        self.null_table = self.local_table + self.nullname
        self.combined_table = comb_name+"_combined" # May vary, so that slightly different combinations can be done side-by-side
        self.full_table_quad = 'all_alternate_quad' # Table of profile likelihood (quadratic approximation) values for all events *and* all alternate hypotheses. Only generated on request due to possibly huge size.
        self.analysis = analysis
        self.null_hyp = null_hyp # Parameters to be used as the null hypothesis
        self.joint = JointDistribution([self.analysis])
        conn = self.connect_to_db()

        # Columns for pseudodata table
        self.event_columns = ["EventID"] # ID number for event, used as index for table  
        for name,eshape in self.analysis.event_shapes().items():
            if eshape==():
                self.event_columns += ["{0}".format(name)]
            elif len(eshape)==1:
                self.event_columns += ["{0}_{1}".format(name,i) for i in range(eshape[0])]
            else:
                msg = "Sorry, LEE routines are currently not compatible with analyses that have an event shape of higher than 1 dimension (which is kind of a weird thing to have, is your analysis constructed correctly? Do you really have a 'tensor' random variable?) Raised due to distribution '{0}' in analysis '{1}' having event_shape = {2})".format(name,analysis.name,eshape)
                raise ValueError(msg)

        #print("columns:", self.event_columns)

        cur = conn.cursor()
        cols = [("EventID", "integer primary key"),("logw", "real")]
        cols += [(col, "real") for col in self.event_columns[1:]]
        event_colnames = [x[0] for x in cols]
        sql.create_table(cur,self.event_table,cols) 
        self.check_table(cur,self.event_table,event_colnames)
        self.close_db(conn)
       
    def check_table(self,cursor,table,required_cols):
        # Print the columns already existing in our table
        cursor.execute('PRAGMA table_info({0})'.format(table))
        results = cursor.fetchall()
        existing_cols = [row[1] for row in results]

        # Check if they match the required columns
        if existing_cols != required_cols:
            msg = "Existing table '{0}' for analyis {1} does not contain the expected columns! May have been created with a different version of this code. Please delete it to recreate the table from scratch.\nExpected columns: {2}\nActual columns:{3}".format(table,self.analysis.name,required_cols,existing_cols)
            raise ValueError(msg) 

    def add_events(self,N,bias=0):
        """Add N new events generated under null hypothesis to the event table.
           Sampling can be biased to higher SR counts with the 'bias' parameter, for
           importance sampling. Make sure to then consider the event weights in final results!"""
        #print("Recording {0} new events...".format(N))
        start = time.time()
        joint = JointDistribution([self.analysis],self.null_hyp)

        # Generate pseudodata
        if bias>0:
            samples, logw = joint.biased_sample(N, bias)
        else:
            samples = joint.sample(N)
            logw = tf.zeros((N,1),dtype=c.TFdtype)

        #print("in add_events: null_hyp:", self.null_hyp)
        #print("in add_events: samples:", samples)

        # Save events to database
        conn = self.connect_to_db()
        cur = conn.cursor()
   
        structure = self.analysis.event_shapes()
        
        command = "INSERT INTO 'events' ('logw'"
        for name, eshape in structure.items():
            if eshape==():
                command += ",`{0}`".format(name)
            elif len(eshape)==1:
                for j in range(eshape[0]):
                    command += ",`{0}_{1}`".format(name,j)
            else:
                msg = "Sorry, LEE routines are currently not compatible with analyses that have an event shape of higher than 1 dimension (which is kind of a weird thing to have, is your analysis constructed correctly? Do you really have a 'tensor' random variable?) Raised due to distribution '{0}' in analysis '{1}' having event_shape = {2})".format(name,self.analysis.name,eshape)
                raise ValueError(msg)
        command += ") VALUES (?"
        for name, size in structure.items():
            if eshape==():
                command += ",?" # Data provided as second argument to 'execute'
            elif len(eshape)==1:
                for j in range(eshape[0]):
                    command += ",?".format(name,j)
            else:
                msg = "Sorry, LEE routines are currently not compatible with analyses that have an event shape of higher than 1 dimension (which is kind of a weird thing to have, is your analysis constructed correctly? Do you really have a 'tensor' random variable?) Raised due to distribution '{0}' in analysis '{1}' having event_shape = {2})".format(name,self.analysis.name,eshape)
                raise ValueError(msg)
        command += ")"         
            
        # Paste together data tables
        datalist = [logw]
        for name in structure.keys():
            sub_events = samples[self.analysis.name+"::"+name]
            datalist += [fix_dims(sub_events)]          
        datatable = tf.concat(datalist,axis=-1).numpy()

        cur.executemany(command,  map(tuple, datatable.tolist())) # sqlite3 doesn't understand numpy types, so need to convert to standard list. Seems fast enough though.
        self.close_db(conn)
        end = time.time()
        #print("Took {0} seconds".format(end-start))
        return logw

    def load_events(self,N=-1,reftable=None,condition=None,offset=0):
        """Loads N (or all by default) events from database where 'condition' is true in 'reftable'
           To skip rows, set 'offset' to the first row to be considered."""
        structure = self.analysis.event_shapes()
        conn = self.connect_to_db()
        cur = conn.cursor()

        if reftable is not None:
            # First see if reference table even exists yes
            if not sql.check_table_exists(cur,reftable):
                condition = None # Ignore conditions if reference table doesn't exist yet. Cannot possible match on them.
            else:
                # See if any data is in the reference table yet:
                cur.execute("SELECT Count(EventID) FROM `{0}`".format(reftable))
                results = cur.fetchall()
                nrows = results[0][0]
                if nrows == 0:
                    # No data; so nothing to match on
                    condition = None
        else:
            # No reference table supplied, so no filter condition can be used.
            condition = None

        command = "SELECT A.EventID"
        for name, eshape in structure.items():
            if eshape==():
                command += ",A.`{0}`".format(name)
            elif len(eshape)==1:
                for j in range(eshape[0]):
                    command += ",A.`{0}_{1}`".format(name,j)
            else:
                msg = "Sorry, LEE routines are currently not compatible with analyses that have an event shape of higher than 1 dimension (which is kind of a weird thing to have, is your analysis constructed correctly? Do you really have a 'tensor' random variable?) Raised due to distribution '{0}' in analysis '{1}' having event_shape = {2})".format(name,self.analysis.name,eshape)
                raise ValueError(msg)

        command += " from events as A"        
        if condition is not None:
           # Apply extra condition to get only events where "neg2LogL" column is NULL (or the EventID doesn't exist) in the 'background' table
           command += """
                      left outer join `{0}` as B
                          on A.EventID=B.EventID
                      where
                          B.{1} 
                      """.format(reftable,condition)
        command += " LIMIT {0} OFFSET {1}".format(N,offset)
        cur.execute(command)
        results = cur.fetchall()
        self.close_db(conn) 

        if len(results)>0:
            # Convert back into dictionary of tensorflow tensors
            # Start by converting to one big tensor
            alldata = tf.convert_to_tensor(results, dtype=c.TFdtype)
 
            EventIDs = tf.cast(tf.round(alldata[:,0]),dtype=tf.int32) # TODO: is this safe?
            i = 1;
            events = {}
            for name, eshape in structure.items():
                if eshape==():
                    size = 1
                elif len(eshape)==1:
                    size = eshape[0]
                else:
                    msg = "Sorry, LEE routines are currently not compatible with analyses that have an event shape of higher than 1 dimension (which is kind of a weird thing to have, is your analysis constructed correctly? Do you really have a 'tensor' random variable?) Raised due to distribution '{0}' in analysis '{1}' having event_shape = {2})".format(name,self.analysis.name,eshape)
                    raise ValueError(msg)

                subevents = tf.expand_dims(alldata[:,i:i+size],axis=1) # insert the 'alternate hypothesis' parameter dimension
                i+=size
                events[self.analysis.name+"::"+name] = subevents

            #print("Retreived eventIDs {0} to {1}".format(np.min(EventIDs.numpy()),np.max(EventIDs.numpy())))
        else:
            EventIDs = None
            events = None
        return EventIDs, events

    def count_events(self):
        """Count number of rows in the event table.
           Should be equal to max(EventID)+1 since we
           don't ever delete events."""
        conn = self.connect_to_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM {0}".format(self.event_table))
        results = cur.fetchall()
        self.close_db(conn) 
        return results[0][0]

    def load_events_with_IDs(self,EventIDs):
        """Loads events with the given eventIDs"""

        if isinstance(EventIDs, str) and EventIDs=='observed':
            events = self.analysis.get_observed_samples()
        else:
            structure = self.analysis.event_shapes()
            cols = []
            for name, eshape in structure.items():
                if eshape==():
                    cols += ["{0}".format(name)]
                elif len(eshape)==1:
                    for j in range(eshape[0]):
                        cols += ["{0}_{1}".format(name,j)]
                else:
                    msg = "Sorry, LEE routines are currently not compatible with analyses that have an event shape of higher than 1 dimension (which is kind of a weird thing to have, is your analysis constructed correctly? Do you really have a 'tensor' random variable?) Raised due to distribution '{0}' in analysis '{1}' having event_shape = {2})".format(name,self.analysis.name,eshape)
                    raise ValueError(msg)

            conn = self.connect_to_db()
            cur = conn.cursor()
            results = sql.load(cur,'events',cols,EventIDs,'EventID')
            self.close_db(conn) 

            if len(results)>0:
                # Convert back into dictionary of tensorflow tensors
                # Start by converting to one big tensor
                alldata = tf.convert_to_tensor(results, dtype=c.TFdtype)
                i = 0;
                events = {}
                for name, eshape in structure.items():
                    if eshape==():
                        size = 1
                    elif len(eshape)==1:
                        size = eshape[0]
                    else:
                        msg = "Sorry, LEE routines are currently not compatible with analyses that have an event shape of higher than 1 dimension (which is kind of a weird thing to have, is your analysis constructed correctly? Do you really have a 'tensor' random variable?) Raised due to distribution '{0}' in analysis '{1}' having event_shape = {2})".format(name,self.analysis.name,eshape)
                        raise ValueError(msg)

                    subevents = tf.expand_dims(alldata[:,i:i+size],axis=1) # insert the 'alternate hypothesis' parameter dimension
                    i+=size
                    events[name] = subevents
            else:
                events = None
        return events

    def load_null_nuis_pars(self,EventIDs):
        """Loads fitted null hypothesis nuisance parameter values for the selected
           events"""
        if isinstance(EventIDs, str) and EventIDs=='observed':
            observed_mode = True
        else:
            observed_mode = False

        nuis_shapes = self.analysis.nuisance_parameter_shapes()
        nuis_indices = c.get_parameter_indices(nuis_shapes)

        cols = []
        for par,shape in nuis_shapes.items():
            for indices in nuis_indices[par]:
                cols += ["{0}::{1}{2}".format(self.analysis.name,par,indices)]

        if len(cols)>0:
            conn = self.connect_to_db()
            cur = conn.cursor()
            if observed_mode:
                results = sql.load(cur,self.null_table+"_observed",cols,[0],'EventID')
            else:
                results = sql.load(cur,self.null_table,cols,EventIDs,'EventID')
            self.close_db(conn) 

            # Convert back to dict of tensors
            if len(results)>0:
                # Convert back into dictionary of tensorflow tensors
                # Start by converting to one big tensor
                alldata = tf.convert_to_tensor(results, dtype=c.TFdtype)
                par_template = {self.analysis.name: nuis_shapes} # The shapes can also be used as an ordering template in this case
                batch_shape = (len(alldata),) # Only ever stored as 1D batch
                nuis_pars = c.decat_tensor_to_pars(alldata,par_template,par_template,batch_shape)
            else:
                nuis_pars = None # No fitted parameters found in output; probably fits weren't done yet. Let calling code decide if this is an error.
        else:
            # No nuisance parameters!
            nuis_pars = {}

        return nuis_pars

    def fit_alternate_batch(self,events,alt_hyp,altIDs=None,record_all=True):
        """Compute alternate hypothesis fits for selected eventIDs
           Returns test statistic values for combination
           with other analyses. 
        """
        # Run full numerical fits of nuisance parameters for all alternate hypotheses
        log_prob_sb, joint_fitted_sb, nuis_pars_s = self.joint.fit_nuisance(events, alt_hyp, log_tag='qsb')

        return log_prob_sb

    def _get_quad(self, EventIDs, expansion_point=None):
        """Get quadratic approximations to likelihood surfaces for the given EventIDs
           (using the null hypothesis (with fitted nuisance parameters) as the expansion point)"""

        if isinstance(EventIDs, str) and EventIDs=='observed':
            observed_mode = True
        else:
            observed_mode = False
            EventIDs = EventIDs.numpy()

        if expansion_point is None:
            # Use null hypothesis with fitted nuisance parameters as expansion point(s).
            # Assume nuisance parameters have already been written to the database and load them
            fitted_nuis_pars = self.load_null_nuis_pars(EventIDs)
            if fitted_nuis_pars is None:
                raise ValueError("Pre-fitted nuisance parameters for a batch of events could not be found! Have all null hypothesis fits been done? Fitted nuisance parameters need to be either directly supplied to this function, or else exist in the output database.")
            expansion_point = c.deep_merge(self.null_hyp, fitted_nuis_pars) 
        events = c.add_prefix(self.analysis.name, self.load_events_with_IDs(EventIDs))
        #print("loaded events:", events)
        quad = self.compute_quad(expansion_point, events)
        return quad

    def compute_quad(self, expansion_pars, events):
        """Compute quadratic approximations of profile likelihood for the specified
           set of events, expanding around the supplied parameter point.

           TODO: Having some shape confusions here. I *think* we need a set of nuisance
           parameters for each event, e.g. the best fit nuisance parameters for each
           event under (say) the background-only hypothesis. So need to make sure pars
           and events supplied have dimensions consistent with this.
        """

        par_size = c.deep_size(expansion_pars,axis=0)

        print("expansion_pars:", expansion_pars)
        print("events:", events)

        # Squeeze out the "hypothesis list" axis from events, to give output Hessians
        # a better shape.
        # TODO: Should only be events for one hypothesis given as input. Check shape
        #events_sq = c.deep_squeeze(events,axis=1)

        #events_sq = {name: fix_dims(e) for name,e in events.items()} 

        #e_size = c.deep_size(events_sq,axis=0)

        #if par_size is None or e_size is None:
        #    raise ValueError("pars or events were empty! pars={0}, events={1}".format(pars,events))

        #if par_size!=-1 and par_size!=e_size:
        #    msg = "Parameters and events did not have sizes consistent for compute_quad: axis 0 size must match! (par_size={0}, e_size={1})".format(par_size,e_size)
        #    raise ValueError(msg)
        ## par_size==-1 is ok, just means there were no nuisance parameters.

        # print("pars:", pars)
        # print("events_sq:", events_sq)

        expansion_point = JointDistribution([self.analysis], expansion_pars)
        quadf = expansion_point.log_prob_quad_f(events)

        #print("quadf:", quadf)
        batch_shape = expansion_point.bcast_batch_shape_tensor()
        print("Expansion point batch_shape:", batch_shape)

        # TODO! This is not the batch shape we want to return!
        # We need the batch shape that we would get if we actually
        # fitted the 'events' (we want the batch shape that would
        # arise from the JointDistribution that would return
        # the same log_probs as log_prob_quad). Need to check the
        # broadcasting rules to remember what this should be!
        # 
        # Side-by-side comparison of calculations (quad vs standard)
        #
        # logp calculation procedure with standard numerical fitting: 
        # 1. Set "signal" evaluation point(s): pars
        # 2. Fit remaining nuisance parameters to samples 
        #    -> output is JointDistribution with all those fitted parameters as the batch_shape
        #    i.e. batch_shape = (samples.shape, pars.shape)
        # 3. Evaluate the log_prob of those same samples
        #    -> output shape is logp.shape = (samples.shape, pars.shape)
        #
        # logp calculation procedure with quad expansion
        # 1. Set "signal" expansion point (just one?) expansion_pars
        # 2. Fit nuisance parameters to samples at that point
        #    -> output is logp_quad_f function
        # 3. Evaluate logp_quad_f for full set of "signal" evaluation points (pars)
        #    -> output shape should be logp.shape = (samples.shape, pars.shape)
        #       (i.e. same as above)
        #
        # But to ensure shapes are correct, we need to know the batch_shape for the JointDistribution
        # from the first method in the second method.
        # ...but we cannot compute it here, because we don't have pars.shape. So we need to provide
        # the rest of the shape information that can then be combined with pars.shape when it becomes
        # available.
        # i.e. need to provide 
        # 

        #raise Exception("Check TODO in the code here!")

        return quadf

    def record_alternate_logLs(self, log_probs, altIDs, EventIDs, Ltype, dbtype='hdf5'):
        """Record likelihoods from a batch of alternate hypothesis fits to 'full' tables.
           In this case ID numbers also need to be assigned to alternate hypotheses,
           so that we can uniquely assign each of them to a row in the
           output table.

           Note: Tables are oriented such that events are columns and alternate hypotheses
           are rows. 1000 events per table, since SQLite likes small numbers 
           of columns (but can hand zillions of rows)
           """
        if Ltype != 'quad':
            raise ValueError("Sorry, alternate hypothesis fit result recording has so far only been implemented for the 'quadratic approximation' results.")

        # Sanity check input shapes
        if len(log_probs.shape) != 2:
            msg = "log_probs supplied for recording are not 2D! Dim 0 should correspond to trials/events/pseudoexperiments/samples, while dim 1 should correspond to 'alternate hypotheses'. Any other dimensions are not valid (e.g. alternate hypothesis arrays must be flattened to 1D). Shape was: {0}".format(log_probs.shape)
            raise ValueError(msg)

        if isinstance(EventIDs, str) and EventIDs=='observed':
            observed_mode = True
        else:
            observed_mode = False
        
        if log_probs.shape[1]>0:
            if dbtype is 'hdf5':
                if observed_mode:
                    fname = '{0}_observed.hdf5'.format(self.db)
                else:
                    fname = '{0}.hdf5'.format(self.db)
                f = h5py.File(fname,'a') # Create if doesn't exist, otherwise read/write
            elif dbtype is 'sqlite':
                conn = self.connect_to_db()
                cur = conn.cursor()
            else:
                raise ValueError("Unrecognised database type selected!")

            # # First split up log_probs into batches to be saved in various of the 'full' tables.
            # # E.g. events in the range 0-999 need to go in table 0, 1000-1999 in table 1, etc.
            # # We will assume that EventIDs already come in ascending order. TODO: Add check for this.
            if observed_mode:
                minTable = 1
                maxTable = 1
                events_per_table = 1
            else:
                events_per_table = 1000
                minEventID = np.min(EventIDs)
                maxEventID = np.max(EventIDs)
                minTable = int(minEventID // events_per_table)
                maxTable = int(maxEventID // events_per_table)

            for i in range(minTable,maxTable+1):
                 if observed_mode:
                     mask = np.array([1],dtype=np.bool)
                 else:
                     this_range = (i*events_per_table+1,(i+1)*events_per_table+1) # EventIDs start at 1
                     mask = (this_range[0] <= EventIDs) & (EventIDs < this_range[1])
                 if np.sum(mask)>0:
                     print("EventIDs:", EventIDs)
                     print("mask:", mask)
                     print("log_probs:", log_probs)
 
                     # BUG HERE
                     # Info: 
                     # mask is supposed to be selecting Events, i.e. pseudo-experiment trials
                     #   i.e. shape (N_trials,)
                     # log_probs is the log-probability of a batch of alternate hypotheses,
                     #   computed via broadcast against trials 
                     #   i.e. shape (N_alt, N_trials)
                     # I think N_alt is always flat? So need to change from this:
                     # log_prob_batch = log_probs[mask]
                     #   to this?
                     # broadcast mask:
                     big_mask = np.array((log_probs*0 + 1) * mask[np.newaxis,:], dtype=np.bool)
                     print("big_mask:", big_mask)
                     log_prob_batch = log_probs[big_mask]

                     if observed_mode:
                         eventID_batch = ['observed']
                     else:
                         eventID_batch = EventIDs[mask]
  
                     # # likelihoods to be stored with one event per column (and zillions of rows corresponding to the alternate hypotheses)
                     # columns = [("E_{0}".format(E_id),"real") for E_id in eventID_batch]
                     # col_names = [x[0] for x in columns]
  
                     # this_table = self.full_table_quad+"_batch_{0}".format(i)
                     # sql.create_table(cur,this_table,[('SignalID',"integer primary key")]+columns)
                     # # If table already existed then we may have to add new event columns
                     # sql.add_columns(cur,this_table,columns)

                     # # Add likelihoods to output record.
                     # data = pd.DataFrame(log_prob_batch.numpy().T,index=altIDs,columns=col_names)
                     # data.index.name = 'SignalID' 
                     # sql.upsert(cur,this_table,data,primary='SignalID')

                     #====== Version 2 ======
                     # Ok it is very slow to retrieve these giant tables with separately stored entries.
                     # However, we *can* just stream raw bits straight into SQL entries. So perhaps just store
                     # entire alternate hypothesis table for each event as one entry. I.e. just one row to retrieve!
                     # Downside is that all alternate hypotheses have to be computed at once, cannot add more.
                     # Or rather more can be added more rows I guess, but won't know if there is overlap.

                     
                     if observed_mode:
                         this_table = self.full_table_quad+'_observed'
                         columns = [("observed",)]
                     else:
                         this_table = self.full_table_quad+"_batch_{0}".format(i)
                         columns = [("E_{0}".format(E_id),"array") for E_id in eventID_batch] # We defined a new datatype, 'array', for sqlite to use to store numpy arrays
                     col_names = [x[0] for x in columns]
                     #print("log_prob_batch:", log_prob_batch.numpy().T)
                     #print("col_names:", col_names)
                     data = pd.DataFrame(log_prob_batch.numpy().T,columns=col_names)

                     if dbtype is 'hdf5':
                         # For hdf5 we don't need to use separate tables, we'll just make one dataset for every event, and extend them as needed.
                         for col in col_names:
                             if col in f.keys():
                                 f[col].resize((len(f[col]) + len(data.index)), axis = 0)
                                 try:
                                     f[col][-len(data.index):] = np.array(data[col])
                                 except TypeError:
                                     print("len(data.index):", len(data.index))
                                     print("np.array(data[col]):", np.array(data[col]))
                                     raise  
                             else:
                                 f.create_dataset(col, data=np.array(data[col]), chunks=(1000,), maxshape=(None,))      

                     elif dbtype is 'sqlite':
                         sql.create_table(cur,this_table,columns)

                         # If table already existed then we may have to add new event columns
                         sql.add_columns(cur,this_table,columns)
                         sql.insert_as_arrays(cur,this_table,data)

            if dbtype is 'hdf5':
                 f.close()
            elif dbtype is 'sqlite':   
                 self.close_db(conn)
        else:
            # Signal dimension is empty! Nothing to record.
            pass
 
    def get_bootstrap_sample(self,N,dbtype='hdf5'):
        """Obtain a bootstrap resampling of size N of the 'full' output tables for all recorded alternate hypotheses"""
        try:
            # If a range of event IDs is passed, just extract those events, don't resample.
            # This is mainly for testing/cross-checking purposes.
            minE,maxE = N
        except TypeError:
            minE, maxE = None, None
            if N==0: raise ValueError("Asked for zero samples!")
            elif N<0: raise ValueError("Asked for negative number of samples!")

        if dbtype is 'hdf5':
            f = h5py.File('{0}.hdf5'.format(self.db),'a') # Create if doesn't exist, otherwise read/write
        elif dbtype is 'sqlite':
            conn = self.connect_to_db()
            cur = conn.cursor()
        else:
            raise ValueError("Unrecognised database type selected!")

        if dbtype is 'sqlite':
            #select * from users
            #where id in (
            #  select round(random() * 21e6)::integer as id
            #  from generate_series(1, 110) -- Preserve duplicates
            #)
            #limit 100
            events_per_table = 1000

            # First inspect full tables to see what the maximum EventID is, so we know what indices to
            # sample from. Assume the set of events is complete up to that maximum number.
            conn = self.connect_to_db()
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            results = cur.fetchall() 
            full_table_batches = [int(row[0].split("_batch_")[1]) for row in results if row[0].startswith(self.full_table_quad)] 
            max_batch = max(full_table_batches)
            max_batch_table = self.full_table_quad+"_batch_{0}".format(max_batch)

            # Inspect the highest batch and find the maximum EventID (column) in it
            cur.execute('PRAGMA table_info({0})'.format(max_batch_table))
            results = cur.fetchall()
            EventIDs = [int(row[1].split("E_")[1]) for row in results if row[1].startswith("E_")]
            maxEventID = max(EventIDs)

        elif dbtype is 'hdf5':
            maxEventID = len(f.keys()) # Assume EventID list is complete           
 
        # Select N integers between 1 and maxEventID with replacement (the bootstrap sample)
        if minE is not None:
            # Just get original events in given range, don't resample
            if minE>maxEventID:
                bootstrap_EventIDs = np.array([],dtype=int)
            else:
                maxE = min([maxE,maxEventID])
                bootstrap_EventIDs = np.arange(minE,maxE+1,dtype=int)
        else:
            #print("maxEventID:", maxEventID)
            #print("N:", N)
            bootstrap_EventIDs = np.random.randint(1,maxEventID,N)
        
        # Sort into ascending order and group according to the batches in which they can be found
        #bootstrap_EventIDs.sort();
        
        # Extract selected bootstrap events table by table. Duplicate column selections should work fine if/when they occur.
        all_events = []
        if dbtype is 'sqlite':
            for i in range(0,max_batch+1):
                this_range = (i*events_per_table+1,(i+1)*events_per_table+1) # EventIDs start at 1
                mask = (this_range[0] <= bootstrap_EventIDs) & (bootstrap_EventIDs < this_range[1])
                if np.sum(mask)>0:
                    these_eventIDs = bootstrap_EventIDs[mask]

                    # Get all logL values for all alternate hypothesis rows for these events
                    command = "SELECT "+", ".join(["E_{0}".format(ID) for ID in these_eventIDs])
                    command += " FROM {0}".format(self.full_table_quad+"_batch_{0}".format(i))
                    #print("command:", command)
                    cur.execute(command)
                    results = cur.fetchall()
                    these_events = []
                    for row in results:
                        these_events += [tf.stack(row,axis=1)] # Join each row of results along events direction 
                    all_events += [tf.concat(these_events,axis=0)] # Join all these events along alternate hypothesis direction
            if len(all_events) > 0:
                alternate_log_prob = tf.concat(all_events,axis=1) # Join all event columns together
            else:
                alternate_log_prob = None
            self.close_db(conn)
        elif dbtype is 'hdf5':
            for ID in bootstrap_EventIDs:
                all_events += [f["E_{0}".format(ID)][:]]  
            f.close()
            if len(all_events) > 0:
                alternate_log_prob = tf.stack(all_events,axis=1) # Join all event columns together
            else:
                alternate_log_prob = None

        # Actually we also need the background-only log_prob values, so grab those too
        # I think here it is fine, and easier, to load them all and do the selection in RAM.
        if len(bootstrap_EventIDs)>0:
            log_p_label = 'log_prob'
            df = self.load_results(self.null_table,['EventID',log_p_label])
            background_log_prob = tf.constant(df.loc[bootstrap_EventIDs][log_p_label].to_numpy(),dtype=c.TFdtype) 
        else:
            background_log_prob = None

        return alternate_log_prob, background_log_prob

    def record_bf_alternate_stats(self):
        pass

    def process_null(self):
        """Compute null-hypothesis fits for events currently in our output tables
           But only for events where this hasn't already been done."""
        self.process_alternate_local()
       
    def process_alternate_local(self,alt_hyp=None,name=None):
        """Compute local hypothesis fits for events currently in our output tables
           But only for events where this hasn't already been done."""
        if alt_hyp is None: 
            alt_hyp = self.null_hyp
            name = self.nullname
            null_case = True
        elif name is None:
            raise ValueError("'name' argument cannot be None for non-null local hypothesis processing! Need a name to identify output in results database")
        else:
            null_case = False

        joint = JointDistribution([self.analysis])

        batch_size = 10000
        continue_processing = True
        total_events = self.count_events()
        bar = Bar("Performing fits of hypothesis '{0}' for analysis {1} in batches of {2} samples".format(name,self.analysis.name,batch_size), max=np.ceil(total_events/batch_size))
        while continue_processing:
            EventIDs, events = self.load_events(batch_size,self.local_table+name,'log_prob is NULL')
            if EventIDs is None:
                # No events left to process
                continue_processing = False
            if continue_processing:
                print("events:", events)
                print("alt_hyp:", alt_hyp)
                log_prob, joint_fitted, nuis_pars = joint.fit_nuisance(events, alt_hyp, log_tag='q_'+name)

                # We need to do some shape adjustment here, because these LEE routines are older than the
                # JointDistribution framework, which has a more general treatment of shapes.
                # Here, we can only deal with 1D batches of parameters, and 1D event shapes. So we need
                # to ensure that only 1D parameter batches are input by the user as alt_hyp, that 
                # scalar events are cast up to 1D, and that higher than 1D event shapes cause explicit
                # errors.
                # So first, check what shapes we are dealing with internally in 'joint' and in the fully
                # fitted joint.
                #print("joint.bcast_batch_shape_tensor():", joint.bcast_batch_shape_tensor()) # Doesn't work 
                print("joint.event_shapes():", joint.event_shapes())
                batch_shape = joint_fitted.bcast_batch_shape_tensor()
                print("joint_fitted.bcast_batch_shape_tensor():", batch_shape) 
                print("joint_fitted.event_shapes():", joint_fitted.event_shapes())

                # Also compute the quad approximation, which can in fact improve on the numerical results since it uses Hessian information. Kind of like a "last step" for the optimizer. Expands around the fitted point, assuming it is close enough to the true best fit for the log-likelihood to be quadratic.
                print("nuis_pars['all']:", nuis_pars['all'])
                quad = self._get_quad(EventIDs,nuis_pars['all'])
                # For testing; only ever expand around null hypothesis, as in profiling case
                #if null_case:
                #    quad = self._get_quad(EventIDs,nuis_pars['all'])
                #else:
                #    quad = self._get_quad(EventIDs)

                log_prob_quad = quad(alt_hyp)
                table = self.local_table+name

                # Check that log_prob_quads are actually better than the original fits
                print("log_prob_quad:", log_prob_quad)
                lpq2D = fix_dims_quad(log_prob_quad, batch_shape)
                lp2D = fix_dims(log_prob)
                m_worse = tf.greater(lp2D, lpq2D)
                n_worse = tf.reduce_sum(tf.cast(m_worse, tf.float32))
                if n_worse>0:
                    print("log_prob:",lp2D)
                    print("log_prob_quad:",lpq2D)
                    print("Some (n={0} out of {1}) quad results are worse than the original fits! local hypothesis = {2}".format(n_worse,m_worse.shape[0],name)) 

                # Write events to output database               
                # Write fitted nuisance parameters to disk as well, for later use in constructing quadratic approximation of profile likelihoods for alternate hypotheses
                # Better to use these rather than parameters that come out of *this* quad function I think...
                arrays = [lp2D, lpq2D]
                cols = ["log_prob", "log_prob_quad"]
                fitted_pars = nuis_pars["fitted"]

                n_pars = len([p for a in fitted_pars.values() for p in a.values()]) 
                if n_pars>0:
                    # Flatten parameters to single 2D tensor
                    par_tensor_2D, batch_shape, flat_par_names = c.cat_pars_to_tensor(fitted_pars,joint.parameter_shapes())
                    arrays += [par_tensor_2D]
                    cols += flat_par_names
                # Else no nuisance parameters! Did no fitting, just evaluated the log_prob directly.

                allpars = tf.concat(arrays,axis=-1)               
                data = pd.DataFrame(allpars.numpy(),index=EventIDs.numpy(),columns=cols)
                data.index.name = 'EventID' 
                conn = self.connect_to_db()
                cur = conn.cursor()
                col_info = [('EventID','integer primary key')] + [(col,'real') for col in cols]
                sql.create_table(cur,self.local_table+name,col_info)
                sql.upsert(cur,self.local_table+name,data,primary='EventID')
                self.close_db(conn)
            bar.next()
        bar.finish()

        if null_case:
            Osamples = joint.Osamples
            # Also record results for fit to observed data
            log_prob_bO, joint_fitted, pars = joint.fit_nuisance(Osamples, self.null_hyp)
            batch_shape = joint_fitted.bcast_batch_shape_tensor()
 
            # Get 'quad' estimate/improvement
            quad = self.compute_quad(pars['all'],Osamples)
            log_prob_quad_O = quad(self.null_hyp)
            # Need the 'effective' batch shape that a JointDistribution fitted 
            # to these samples using these parameters would have. 
            eff_batch_shape = joint.expected_batch_shape_nuis(pars['all'], Osamples)        
            print("effective batch_shape:", eff_batch_shape)

            arrays = [fix_dims(log_prob_bO),fix_dims_quad(log_prob_quad_O,eff_batch_shape)]
            cols = ['log_prob','log_prob_quad']
            fitted_pars = pars["fitted"]

            n_pars = len([p for a in fitted_pars.values() for p in a.values()]) 
            if n_pars>0:
                # Flatten parameters to single 2D tensor
                par_tensor_2D, batch_shape, flat_par_names = c.cat_pars_to_tensor(fitted_pars,joint.parameter_shapes())
                arrays += [par_tensor_2D]
                cols += flat_par_names
            # Else no nuisance parameters! Did no fitting, just evaluated the log_prob directly.

            print("arrays:", arrays)
            allpars = tf.concat(arrays,axis=-1)               
            data = pd.DataFrame(allpars.numpy(),columns=cols)
            data.index.rename('EventID',inplace=True) 
            conn = self.connect_to_db()
            cur = conn.cursor()
            col_info = [('EventID','integer primary key')] + [(col,'real') for col in cols]
            sql.create_table(cur,self.local_table+name+"_observed",col_info)
            sql.upsert(cur,self.local_table+name+"_observed",data,primary='EventID')
            self.close_db(conn)

        else:
            # If this isn't the null case, then should also compute the various Asimov likelihoods for this hypothesis, for asymptotic results
            joint0s = JointDistribution([self.analysis],c.deep_expand_dims(alt_hyp,axis=0)) # dims expanded so that hypothesis treated as a list of hypotheses, currently required by compute_quad function.
            joint0  = JointDistribution([self.analysis],c.deep_expand_dims(self.null_hyp,axis=0))
 
            # Get Asimov samples for nuisance MLEs with fixed alternate hypotheses
            samplesAsb = joint0s.Asamples
            samplesAb  = joint0.Asamples

            # Evaluate distributions for Asimov datasets, in case where we
            # know that the MLEs for those samples are the true parameters
            # (i.e. don't have to fit because we just recover the parameters that we put in)
            log_prob_sbAsb = joint0s.log_prob(samplesAsb)
            log_prob_bAb  = joint0.log_prob(samplesAb)
            
            # Fit distributions for Asimov datasets for the other half of each
            # likelihood ratio
            log_prob_bAsb, joint_fitted, pars_Asb = joint.fit_nuisance(samplesAsb, self.null_hyp, log_tag='bAsb')
            log_prob_sbAb, joint_fitted, pars_Ab = joint.fit_nuisance(samplesAb, alt_hyp, log_tag='sbAsb')

            qAsb = -2*(log_prob_sbAsb - log_prob_bAsb)[0] # extract single sample result
            qAb = -2*(log_prob_sbAb - log_prob_bAb)[0]
 
            # Get 'quad' estimate/improvements
            quad_Asb = self.compute_quad(pars_Asb['all'],samplesAsb)
            quad_Ab  = self.compute_quad(pars_Ab['all'], samplesAb)
            log_prob_quad_bAsb = quad_Asb(self.null_hyp)
            log_prob_quad_sbAb = quad_Ab(alt_hyp)

            qAsb_quad = -2*(log_prob_sbAsb - log_prob_quad_bAsb)[0] # extract single sample result
            qAb_quad = -2*(log_prob_quad_sbAb - log_prob_bAb)[0]
 
            # ...and fit the observed data too!
            Osamples = joint.Osamples
            log_prob_bO , joint_fitted, pars_bO  = joint.fit_nuisance(Osamples, self.null_hyp)
            log_prob_sbO, joint_fitted, pars_sbO = joint.fit_nuisance(Osamples, alt_hyp)
            qO = -2*(log_prob_sbO - log_prob_bO)[0] # extract single sample result

            # ...and quad improved version of that too
            quad_bO  = self.compute_quad(pars_bO['all'],Osamples)
            quad_sbO = self.compute_quad(pars_sbO['all'],Osamples)
            log_prob_quad_bO  = quad_bO(self.null_hyp)
            log_prob_quad_sbO = quad_sbO(alt_hyp)
            qO_quad = -2*(log_prob_quad_sbO - log_prob_quad_bO)[0] # extract single sample result

            # Record everything!
            cols = ['qAsb','qAb','qO','log_prob']
            cols_quad = ['qAsb_quad','qAb_quad','qO_quad','log_prob_quad']
            d_in = [qAsb,qAb,qO,log_prob_sbO]
            d_in_quad = [qAsb_quad,qAb_quad,qO_quad,log_prob_quad_sbO]
            all_d = d_in+d_in_quad
            all_cols = cols+cols_quad
            d = np.array([v.numpy() for v in all_d]) 
            data = pd.DataFrame(d.T,columns=all_cols)
            data.index.rename('EventID',inplace=True)
            #print("data:",data)
            conn = self.connect_to_db()
            cur = conn.cursor()
            col_info = [('EventID','integer primary key')]+[(col,'real') for col in all_cols]
            sql.create_table(cur,self.local_table+name+'_observed',col_info)
            sql.upsert(cur,self.local_table+name+'_observed',data,primary='EventID')
            self.close_db(conn)


def collider_analyses_from_long_YAML(yamlfile,replace_SR_names=False):
    """Read long format YAML file of analysis data into BinnedAnalysis objects"""
    #print("Loading YAML")
    d = yaml.safe_load(yamlfile)
    #print("YAML loaded")
    analyses = []
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
        a = BinnedAnalysis(k,srs,cov,cov_order,ucz)
        analyses += [a]
    if replace_SR_names:
        return analyses, SR_name_translation, inverse_translation
    else:
        return analyses
