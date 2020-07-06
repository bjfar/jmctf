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
from . import sql_helpers as sql
from . import common as com
from .binned_analysis import BinnedAnalysis
from .joint import JointDistribution

def LEEcorrection(analyses,signal,nosignal,name,N,fitall=True):
    """Compute LEE-corrected p-value to exclude the no-signal hypothesis, with
       'signal' providing the set of signals to consider for the correction
       
       Also computes local p-values for all input signal hypotheses.
    """
    nullnuis = {a.name: {'nuisance': None} for a in analyses.values()} # Use to automatically set nuisance parameters to zero for sample generation

    print("Simulating {0}".format(name))
    
    # Create joint distributions
    joint   = JointDistribution(analyses) # Version for fitting (object is left with fitted parameters upon fitting)
    joint0  = JointDistribution(analyses,com.deep_merge(nosignal,nullnuis))
    joint0s = JointDistribution(analyses,com.deep_merge(signal,nullnuis))
    
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
        c = conn.cursor()
        results = sql.load(c,table,columns,keys,primary)
        info = sql.table_info(c,table)

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
       for large numbers of analyses, with many signal hypotheses to test, and with many
       random MC draws. Allows more signal hypotheses and random draws to be added to the
       simulation without recomputing everything"""

    def __init__(self,analyses,path,master_name,nullsignals,nullname):
        super().__init__(path,"{0}_{1}".format(master_name,nullname))

        self.LEEanalyses = {}
        self.profiled_table = 'profiled' # For combined results profiled over all signal hypotheses
        self.nullname = 'null' # For combined results for null hypothesis
        self.local_table = 'local_' # Base name for tables related to specific signal hypotheses whose local properties we want to investigate
        for name,a in analyses.items():
            self.LEEanalyses[name] = LEECorrectorAnalysis(a,path,master_name,{a.name: nullsignals[a.name]},nullname)

        conn = self.connect_to_db()
        c = conn.cursor()

        # Table for recording final profiled (i.e. best-fit) test statistic results
        comb_cols = [ ("EventID", "integer primary key")
                     ,("neg2logL", "real")
                     ,("neg2logL_quad", "real")
                     ,("logw", "real")]
        colnames = [x[0] for x in comb_cols]

        sql.create_table(c,self.profiled_table,comb_cols)
        sql.create_table(c,self.profiled_table+"_observed",comb_cols[:-1]) # Don't want the logw column for this
        self.close_db(conn)
  
    def add_events(self,N,bias=0):
        """Generate pseudodata for all analyses under the null signal hypothesis"""
        logw = tf.zeros((N,1))
        for name,a in self.LEEanalyses.items():
            logw += a.add_events(N,bias)
        # Record the eventIDs in the combined database, plus the combined weights
        self.add_events_comb(N,logw)

    def add_events_comb(self,N,logw):
        """Record existence of events in the profiled table"""
        # Do this by just inserting null data and letting the EventID column auto-increment
        conn = self.connect_to_db()
        c = conn.cursor()
        command = "INSERT INTO {0} (neg2logL,logw) VALUES (?,?)".format(self.profiled_table)
        vals = [(None,logwi[0]) for logwi in logw.numpy().tolist()]
        c.executemany(command,  vals) 
        self.close_db(conn)

    def process_null(self):
        """Perform nuisance parameter fits of null hypothesis for all events where it hasn't already been done"""
        self.process_signal_local() # Special case of local signal processing for the null signal case.

    def process_signal_local(self,signal=None,name=None):
        """Perform nuisance parameter fits of null hypothesis for all events where it hasn't already been done"""
        for a in self.LEEanalyses.values():
            if signal is None:
                a.process_null()
            elif name is None:
                raise ValueError("Name for local signal needs to be provided, for identifying it in the results database!")
            else:
                a.process_signal_local(signal,name)
   
        # Fits completed; extract all results and get combined neg2logL values
        comb = None
        for a in self.LEEanalyses.values():
            if signal is None: table = a.local_table+a.nullname
            else: table = a.local_table+name
            df = a.load_results(table,['EventID','neg2logL'])
            #print("df:",df)
            if comb is None: 
                comb = df
            else:
                comb += df
        conn = self.connect_to_db()
        c = conn.cursor()
        if signal is None: table = self.local_table+self.nullname
        else: table = self.local_table+name
        sql.create_table(c,table,[('EventID','integer primary key'),('neg2logL','real')])
        sql.upsert(c,table,comb,'EventID')
        self.close_db(conn)

        # Do the same for asymptotic/Asimov and observed values (non-null case only)
        if signal is not None:
            comb = None
            cols = ['qAsb','qAb','qO','neg2logL','neg2logL_quad']
            for a in self.LEEanalyses.values():
                table = a.local_table+name+"_observed"
                df = a.load_results(table,cols)
                if comb is None: 
                    comb = df
                else:
                    comb += df
            comb.index.rename('EventID',inplace=True)
            conn = self.connect_to_db()
            c = conn.cursor()
            col_info = [('EventID','integer primary key')]+[(col,'real') for col in cols]
            sql.create_table(c,self.local_table+name+'_observed',col_info)
            sql.upsert(c,self.local_table+name+'_observed',comb,primary='EventID')
            self.close_db(conn)
        else:
            # Else just add the observed value for the null hypothesis likelihood
            comb = None
            cols = ['neg2logL'] # We never use the quad approximation for the null likelihood, since this is the point we expand around.
            for a in self.LEEanalyses.values():
                table = a.local_table+a.nullname+"_observed"
                df = a.load_results(table,cols)
                if comb is None: 
                    comb = df
                else:
                    comb += df
            comb.index.rename('EventID',inplace=True)
            conn = self.connect_to_db()
            c = conn.cursor()
            col_info = [('EventID','integer primary key')]+[(col,'real') for col in cols]
            sql.create_table(c,self.local_table+self.nullname+'_observed',col_info)
            sql.upsert(c,self.local_table+self.nullname+'_observed',comb,primary='EventID')
            self.close_db(conn)


    def _process_signal_batch(self,signal_gen,EventIDs,dbtype):
        """For internal use in 'process_signals' function. Processes a single batch of events."""
        quads = []

        if isinstance(EventIDs, str) and EventIDs=='observed':
            observed_mode = True
        else:
            observed_mode = False
            EventIDs = EventIDs.numpy()

        for name,a in self.LEEanalyses.items():
            pars = a.load_bg_nuis_pars(EventIDs)
            if pars is None:
                raise ValueError("Pre-fitted nuisance parameters for a batch of events could not be found! Have all background fits been done?")
            events = a.load_events(EventIDs)
            quads += [a.compute_quad(pars,events)]
        Ns = signal_gen.count
        Nchunk = signal_gen.chunk_size
        Nbatches = Ns // Nchunk
        rem = Ns % Nchunk
        if rem!=0: Nbatches+=1
        j=0
        bar = Bar('Processing signals in batches of {0}'.format(Nchunk), max=Nbatches)
        signal_gen.reset()
        min_neg2logLs = None
        for i in range(Nbatches):
            if rem!=0 and i==Nbatches: size = rem
            else: size = Nchunk
            comb_neg2logLs = None
            sig_chunk, signalIDs = signal_gen.next() #{name: {par: dat[j:j+size] for par,dat in signals[name].items()}}
            for quad,(name,a) in zip(quads,self.LEEanalyses.items()):
                #print("running quad:",name)
                neg2logLs = quad({name: sig_chunk[name]})
                # Record all signal likelihoods to disk, so that we can use them for bootstrap resampling later on.
                # Warning: may take a lot of disk space if there are a lot of signals.
                a.record_signal_logLs(neg2logLs,signalIDs,EventIDs,Ltype='quad',dbtype=dbtype)
                #print("...done")
                #print("signal neg2logLs:", neg2logLs)
                if comb_neg2logLs is None:
                    comb_neg2logLs = neg2logLs
                else:
                    comb_neg2logLs += neg2logLs

            #print("sig_chunk:", sig_chunk)
            # Select the mininum -2logL from across all signal hypotheses
            if min_neg2logLs is not None:
                all_neg2logLs = tf.concat([tf.expand_dims(min_neg2logLs,axis=-1),comb_neg2logLs],axis=-1)
            else:
                all_neg2logLs = comb_neg2logLs
            min_neg2logLs = tf.reduce_min(all_neg2logLs,axis=-1)
            j += size 
            bar.next()
        bar.finish() 
        return min_neg2logLs  
 
    def process_signals_observed(self,signal_gen,quad_only=True,dbtype='hdf5'):
        """Perform fits for all supplied signal hypotheses, for just the *observed* data"""
        min_neg2logLs = self._process_signal_batch(signal_gen,"observed",dbtype)
        
        # Extract data for null hypothesis (should be pre-computed by process_null)

              
        # Write the compute min_neg2logLs to disk for this batch of events
        data = pd.DataFrame(min_neg2logLs.numpy(),columns=['neg2logL_quad'])
        data.index.name = 'EventID' 
 
        conn = self.connect_to_db()
        c = conn.cursor()
        sql.upsert_if_smaller(c,self.profiled_table+"_observed",data,'EventID') # Only replace existing values if new ones are smaller
        self.close_db(conn)


    def process_signals(self,signal_gen,quad_only=True,new_events_only=False,event_batch_size=1000,dbtype='hdf5'):
        """Perform fits for all supplied signal hypotheses, for all events in the database,
           and record the best-fit signal for each event. Compares to any existing best-fits
           in the database and updates if the new best-fit is better.
           If 'new_events_only' is true, fits are only performed for events in the database
           for which no signal fit results are yet recorded."""
        # First process the observed signal
        print("Processing signals for *observed* dataset")
        self.process_signals_observed(signal_gen,quad_only,dbtype)
        print("...done!")
        Nevents = self.count_events_comb()
        still_processing = True
        offset = 0
        N_event_batches = int(np.ceil(Nevents / event_batch_size))
        batchi = 1; 
        while still_processing:
            print("Processing event batch {0} of {1} (batch_size={2})".format(batchi,N_event_batches,event_batch_size))
            if new_events_only:
                EventIDs = self.load_eventIDs(event_batch_size,self.profiled_table,'neg2logL_quad is NULL')
            else:
                EventIDs = self.load_eventIDs(event_batch_size,offset=offset)
                offset += event_batch_size
            if EventIDs is None: still_processing = False
            if still_processing:
                min_neg2logLs = self._process_signal_batch(signal_gen,EventIDs,dbtype)              
                # Write the compute min_neg2logLs to disk for this batch of events
                data = pd.DataFrame(min_neg2logLs.numpy(),index=EventIDs.numpy(),columns=['neg2logL_quad'])
                data.index.name = 'EventID' 
 
                conn = self.connect_to_db()
                c = conn.cursor()
                sql.upsert_if_smaller(c,self.profiled_table,data,'EventID') # Only replace existing values if new ones are smaller
                self.close_db(conn)

                batchi+=1

    def get_bootstrap_sample(self,N,batch_size=2000,dbtype='hdf5'):
        """Obtain a bootstrap resampling of all 'full' tables in all analyses, 
           combine/profile the likelihoods, and add results to bootstrap table.

           Keep batch size small when number of signals is large, to avoid
           running out of RAM. Will have likelihoods for all bootstrap events
           for all signals."""
  
        all_min_neg2logL = None
        all_b_neg2logL = None
        if N=='all': # Special keyword to just re-do profiling rather than bootstrap resampling. To cross-check this profiling calculation with original calculation.
            print("Recomputing profile over signal hypothesis for existing events (no resampling...)")
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
                all_neg2logLs = None
                these_b_neg2logLs = None
                for name,a in self.LEEanalyses.items():
                    s_neg2logLs, b_neg2logLs = a.get_bootstrap_sample(size,dbtype=dbtype)
                    if s_neg2logLs is None: 
                        done = True
                        break
                    #print("s_neg2logLs.shape:", s_neg2logLs.shape)
                    #print("b_neg2logLs.shape:", b_neg2logLs.shape)
                    if all_neg2logLs is None: all_neg2logLs = s_neg2logLs
                    else: all_neg2logLs += s_neg2logLs
                    if these_b_neg2logLs is None: these_b_neg2logLs = b_neg2logLs
                    else: these_b_neg2logLs += b_neg2logLs
                    if N=='all':
                        print("Analysis {0}, batch {1}".format(name, i))
                    else:
                        bar.next()


                if not done:
                    #print("all_neg2logLs.shape:",all_neg2logLs.shape)

                    # Profile
                    min_neg2logL = tf.reduce_min(all_neg2logLs,axis=0) # Signal dimension is first here, different to elsewhere
                    #print("min_neg2logL.shape:", min_neg2logL.shape)
                            
                    # Write to disk? Could just return if this is fast. Test to find out.
                    if all_min_neg2logL is None: all_min_neg2logL = min_neg2logL
                    else: all_min_neg2logL = tf.concat([all_min_neg2logL,min_neg2logL],axis=0) # Only one dimension left
 
                    if all_b_neg2logL is None: all_b_neg2logL = these_b_neg2logLs
                    else: all_b_neg2logL = tf.concat([all_b_neg2logL,these_b_neg2logLs],axis=0)
            i+=1
        if N!='all': bar.finish()
        return all_min_neg2logL, all_b_neg2logL

    def load_eventIDs(self,N,reftable=None,condition=None,offset=0):
        """Loads eventIDs from database where 'condition' is true in 'reftable'
           To skip rows, set 'offset' to the first row to be considered."""
        conn = self.connect_to_db()
        c = conn.cursor()

        if reftable is not None:
            # First see if any data is in the reference table yet:
            c.execute("SELECT Count(EventID) FROM "+reftable)
            results = c.fetchall()
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
        c.execute(command)
        results = c.fetchall()
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
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM {0}".format(self.profiled_table))
        results = c.fetchall()
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

    def add_signals(self,signals):
        pass

    def add_to_database(self):
        pass

    def compute_quad(self,data):
        """Compute quadratic approximations of profile likelihood for a set of
           data"""
        pass

class LEECorrectorAnalysis(LEECorrectorBase):
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

    def __init__(self,analysis,path,comb_name,nullsignal,nullname):
        super().__init__(path,analysis.name)

        self.event_table = 'events'
        self.nullname = 'null' # String to identify null hypothesis tables
        self.local_table = 'local_' # Base name for tables containing local TS data)
        self.combined_table = comb_name+"_combined" # May vary, so that slightly different combinations can be done side-by-side
        self.full_table_quad = 'all_signals_quad' # Table of profile likelihood (quadratic approximation) values for all events *and* all signals. Only generated on request due to possibly huge size.
        self.analysis = analysis
        self.analyses = {self.analysis.name: self.analysis} # For functions that expect a dictionary of analyses
        self.nullsignal = nullsignal # "signal" parameters to be used as the 'background-only' null hypothesis
        self.nullnuis = {self.analysis.name: {'nuisance': None}} # Use to automatically set nuisance parameters to zero for sample generation
        self.joint = JointDistribution(self.analyses)
        conn = self.connect_to_db()

        # Columns for pseudodata table
        self.event_columns = ["EventID"] # ID number for event, used as index for table  
        for name,dim in self.analysis.get_sample_structure().items():
           self.event_columns += ["{0}_{1}".format(name,i) for i in range(dim)]

        #print("columns:", self.event_columns)

        c = conn.cursor()
        cols = [("EventID", "integer primary key"),("logw", "real")]
        cols += [(col, "real") for col in self.event_columns[1:]]
        event_colnames = [x[0] for x in cols]
        sql.create_table(c,self.event_table,cols) 
        self.check_table(c,self.event_table,event_colnames)
        self.close_db(conn)
       
    def check_table(self,c,table,required_cols):
        # Print the columns already existing in our table
        c.execute('PRAGMA table_info({0})'.format(table))
        results = c.fetchall()
        existing_cols = [row[1] for row in results]

        # Check if they match the required columns
        if existing_cols != required_cols:
            msg = "Existing table '{0}' for analyis {1} does not contain the expected columns! May have been created with a different version of this code. Please delete it to recreate the table from scratch.\nExpected columns: {2}\nActual columns:{3}".format(table,self.analysis.name,required_cols,existing_cols)
            raise ValueError(msg) 

    def add_events(self,N,bias=0):
        """Add N new events generated under null hypothesis signal to the event table.
           Sampling can be biased to higher SR counts with the 'bias' parameter, for
           importance sampling. Make sure to then consider the event weights in final results!"""
        #print("Recording {0} new events...".format(N))
        start = time.time()
        joint = JointDistribution(self.analyses,com.deep_merge(self.nullsignal,self.nullnuis))

        # Generate pseudodata
        if bias>0:
            samples, logw = joint.biased_sample(N, bias)
        else:
            samples = joint.sample(N)
            logw = tf.zeros((N,1),dtype=c.TFdtype)

        # Save events to database
        conn = self.connect_to_db()
        c = conn.cursor()
   
        structure = self.analysis.get_sample_structure() 
        
        command = "INSERT INTO 'events' ('logw'"
        for name, size in structure.items():
            for j in range(size):
                command += ",'{0}_{1}'".format(name,j)
        command += ") VALUES (?"
        for name, size in structure.items():
            for j in range(size):
                command += ",?" # Data provided as second argument to 'execute'
        command += ")"         
            
        # Paste together data tables
        datalist = [logw]
        for name, size in structure.items():
            datalist += [tf.squeeze(samples[self.analysis.name+"::"+name],axis=1)] # Remove 'signal' dimension, should be size 1 (TODO: add check for this)
         
        datatable = tf.concat(datalist,axis=-1).numpy()

        c.executemany(command,  map(tuple, datatable.tolist())) # sqlite3 doesn't understand numpy types, so need to convert to standard list. Seems fast enough though.
        self.close_db(conn)
        end = time.time()
        #print("Took {0} seconds".format(end-start))
        return logw

    def load_N_events(self,N,reftable,condition=None,offset=0):
        """Loads N events from database where 'condition' is true in 'reftable'
           To skip rows, set 'offset' to the first row to be considered."""
        structure = self.analysis.get_sample_structure() 
        conn = self.connect_to_db()
        c = conn.cursor()

        # First see if reference table even exists yes
        if not sql.check_table_exists(c,reftable):
            condition = None # Ignore conditions if reference table doesn't exist yet. Cannot possible match on them.
        else:
            # See if any data is in the reference table yet:
            c.execute("SELECT Count(EventID) FROM "+reftable)
            results = c.fetchall()
            nrows = results[0][0]
            if nrows == 0:
                # No data; so nothing to match on
                condition = None

        command = "SELECT A.EventID"
        for name, size in structure.items():
            for j in range(size):
                command += ",A.{0}_{1}".format(name,j)

        command += " from events as A"        
        if condition is not None:
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
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM {0}".format(self.event_table))
        results = c.fetchall()
        self.close_db(conn) 
        return results[0][0]

    def load_events(self,EventIDs):
        """Loads events with the given eventIDs"""

        if isinstance(EventIDs, str) and EventIDs=='observed':
            events = self.analysis.get_observed_samples()
        else:
            structure = self.analysis.get_sample_structure() 
            cols = []
            for name, size in structure.items():
                for j in range(size):
                    cols += ["{0}_{1}".format(name,j)]

            conn = self.connect_to_db()
            c = conn.cursor()
            results = sql.load(c,'events',cols,EventIDs,'EventID')
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
        if isinstance(EventIDs, str) and EventIDs=='observed':
            observed_mode = True
        else:
            observed_mode = False

        nuis_structure = self.analysis.get_nuisance_parameter_structure()
        cols = []
        for par,size in nuis_structure.items():
            for i in range(size):
                cols += ["{0}_{1}".format(par,i)]

        conn = self.connect_to_db()
        c = conn.cursor()
        if observed_mode:
            results = sql.load(c,self.local_table+self.nullname+"_observed",cols,[0],'EventID')
        else:
            results = sql.load(c,self.local_table+self.nullname,cols,EventIDs,'EventID')
        self.close_db(conn) 

        # Convert back to dict of tensors
        if len(results)>0:
            # Convert back into dictionary of tensorflow tensors
            # Start by converting to one big tensor
            alldata = tf.convert_to_tensor(results, dtype=tf.float32)
            i = 0
            pars = {}
            for par,size in nuis_structure.items():
                pars[par] = tf.expand_dims(alldata[:,i:i+size],axis=1) # insert the 'signal' parameter dimension
                i+=size
            nuis_pars = {self.analysis.name: pars}
        else:
            nuis_pars = None
        return nuis_pars
 
    def fit_signal_batch(self,events,signals,signalIDs=None,record_all=True):
        """Compute signal fits for selected eventIDs
           Returns test statistic values for combination
           with other analyses. 
        """
        # Run full numerical fits of nuisance parameters for all signal hypotheses
        qsb, joint_fitted_sb, nuis_pars_s = self.joint.fit_nuisance(signals, events, log_tag='qsb')

        return qsb

    def compute_quad(self,pars,events):
        """Compute quadratic approximations of profile likelihood for the specified
           set of events, expanding around the supplied nuisance parameter point with
           the null signal"""
        joint_fitted_b = JointDistribution(self.analyses,com.deep_merge(self.nullsignal,pars))
        quadf = joint_fitted_b.quad_loglike_f(events)

        return quadf 

    def record_signal_logLs(self,neg2logLs,signalIDs,EventIDs,Ltype,dbtype='hdf5'):
        """Record likelihoods from a batch of signal fits to 'full' tables.
           In this case ID numbers also need to be assigned to signals,
           so that we can uniquely assign each of them to a row in the
           output table.

           Note: Tables are oriented such that events are columns and signals
           are rows. 1000 events per table, since SQLite likes small numbers 
           of columns (but can hand zillions of rows)
           """
        if Ltype != 'quad':
            raise ValueError("Sorry, signal fit result recording has so far only been implemented for the 'quadratic approximation' results.")

        if isinstance(EventIDs, str) and EventIDs=='observed':
            observed_mode = True
        else:
            observed_mode = False
        
        if neg2logLs.shape[1]>0:
            if dbtype is 'hdf5':
                if observed_mode:
                    fname = '{0}_observed.hdf5'.format(self.db)
                else:
                    fname = '{0}.hdf5'.format(self.db)
                f = h5py.File(fname,'a') # Create if doesn't exist, otherwise read/write
            elif dbtype is 'sqlite':
                conn = self.connect_to_db()
                c = conn.cursor()
            else:
                raise ValueError("Unrecognised database type selected!")

            # # First split up neg2logLs into batches to be saved in various of the 'full' tables.
            # # E.g. events in the range 0-999 need to go in table 0, 1000-1999 in table 1, etc.
            # # We will assume that EventIDs already come in ascending order. TODO: Add check for this.
            if observed_mode:
                minTable = 1
                maxTable = 1
                events_per_table = 1
            else:
                events_per_table = 1000.
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
                     neg2logL_batch = neg2logLs[mask]
                     if observed_mode:
                         eventID_batch = ['observed']
                     else:
                         eventID_batch = EventIDs[mask]
  
                     # # likelihoods to be stored with one event per column (and zillions of rows corresponding to the signals)
                     # columns = [("E_{0}".format(E_id),"real") for E_id in eventID_batch]
                     # col_names = [x[0] for x in columns]
  
                     # this_table = self.full_table_quad+"_batch_{0}".format(i)
                     # sql.create_table(c,this_table,[('SignalID',"integer primary key")]+columns)
                     # # If table already existed then we may have to add new event columns
                     # sql.add_columns(c,this_table,columns)

                     # # Add likelihoods to output record.
                     # data = pd.DataFrame(neg2logL_batch.numpy().T,index=signalIDs,columns=col_names)
                     # data.index.name = 'SignalID' 
                     # sql.upsert(c,this_table,data,primary='SignalID')

                     #====== Version 2 ======
                     # Ok it is very slow to retrieve these giant tables with separately stored entries.
                     # However, we *can* just stream raw bits straight into SQL entries. So perhaps just store
                     # entire signal table for each event as one entry. I.e. just one row to retrieve!
                     # Downside is that all signals have to be computed at once, cannot add more.
                     # Or rather more can be added more rows I guess, but won't know if there is overlap.

                     
                     if observed_mode:
                         this_table = self.full_table_quad+'_observed'
                         columns = [("observed",)]
                     else:
                         this_table = self.full_table_quad+"_batch_{0}".format(i)
                         columns = [("E_{0}".format(E_id),"array") for E_id in eventID_batch] # We defined a new datatype, 'array', for sqlite to use to store numpy arrays
                     col_names = [x[0] for x in columns]
                     data = pd.DataFrame(neg2logL_batch.numpy().T,columns=col_names)

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
                         sql.create_table(c,this_table,columns)

                         # If table already existed then we may have to add new event columns
                         sql.add_columns(c,this_table,columns)
                         sql.insert_as_arrays(c,this_table,data)

            if dbtype is 'hdf5':
                 f.close()
            elif dbtype is 'sqlite':   
                 self.close_db(conn)
        else:
            # Signal dimension is empty! Nothing to record.
            pass
 
    def get_bootstrap_sample(self,N,dbtype='hdf5'):
        """Obtain a bootstrap resampling of size N of the 'full' output tables for all recorded signals"""
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
            c = conn.cursor()
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
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            results = c.fetchall() 
            full_table_batches = [int(row[0].split("_batch_")[1]) for row in results if row[0].startswith(self.full_table_quad)] 
            max_batch = max(full_table_batches)
            max_batch_table = self.full_table_quad+"_batch_{0}".format(max_batch)

            # Inspect the highest batch and find the maximum EventID (column) in it
            c.execute('PRAGMA table_info({0})'.format(max_batch_table))
            results = c.fetchall()
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

                    # Get all logL values for all signal rows for these events
                    command = "SELECT "+", ".join(["E_{0}".format(ID) for ID in these_eventIDs])
                    command += " FROM {0}".format(self.full_table_quad+"_batch_{0}".format(i))
                    #print("command:", command)
                    c.execute(command)
                    results = c.fetchall()
                    these_events = []
                    for row in results:
                        these_events += [tf.stack(row,axis=1)] # Join each row of results along events direction 
                    all_events += [tf.concat(these_events,axis=0)] # Join all these events along signal direction
            if len(all_events) > 0:
                signal_neg2logL = tf.concat(all_events,axis=1) # Join all event columns together
            else:
                signal_neg2logL = None
            self.close_db(conn)
        elif dbtype is 'hdf5':
            for ID in bootstrap_EventIDs:
                all_events += [f["E_{0}".format(ID)][:]]  
            f.close()
            if len(all_events) > 0:
                signal_neg2logL = tf.stack(all_events,axis=1) # Join all event columns together
            else:
                signal_neg2logL = None

        # Actually we also need the background-only neg2logL values, so grab those too
        # I think here it is fine, and easier, to load them all and do the selection in RAM.
        if len(bootstrap_EventIDs)>0:
            df = self.load_results(self.local_table+self.nullname,['EventID','neg2logL'])
            background_neg2logL = tf.constant(df.loc[bootstrap_EventIDs]['neg2logL'].to_numpy(),dtype=c.TFdtype) 
        else:
            background_neg2logL = None

        return signal_neg2logL, background_neg2logL

    def record_bf_signal_stats(self):
        pass

    def process_null(self):
        """Compute null-hypothesis fits for events currently in our output tables
           But only for events where this hasn't already been done."""
        self.process_signal_local()
       
    def process_signal_local(self,signal=None,name=None):
        """Compute local hypothesis fits for events currently in our output tables
           But only for events where this hasn't already been done."""
        if signal is None: 
            signal = self.nullsignal
            name = self.nullname
            null_case = True
        elif name is None:
            raise ValueError("'name' argument cannot be None for non-null local signal processing! Need a name to identify output in results database")
        else:
            null_case = False

        joint = JointDistribution(self.analyses)

        batch_size = 10000
        continue_processing = True
        total_events = self.count_events()
        bar = Bar("Performing fits of signal '{0}' for analysis {1} in batches of {2}".format(name,self.analysis.name,batch_size), max=np.ceil(total_events/batch_size))
        while continue_processing:
            EventIDs, events = self.load_N_events(batch_size,self.local_table+name,'neg2logL is NULL')
            if EventIDs is None:
                # No events left to process
                continue_processing = False
            if continue_processing:
                #print("Fitting w.r.t background-only samples")
                qb, joint_fitted_b, nuis_pars_b = joint.fit_nuisance(signal, events, log_tag='q_'+name)
                #print("nuis_pars_b:", nuis_pars_b)
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
                col_info = [('EventID','integer primary key')] + [(col,'real') for col in cols]
                sql.create_table(c,self.local_table+name,col_info)
                sql.upsert(c,self.local_table+name,data,primary='EventID')
                self.close_db(conn)
            bar.next()
        bar.finish()

        if null_case:
            Osamples = joint.Osamples
            # Also record results for fit to observed data
            qbO, joint_fitted, pars = joint.fit_nuisance(self.nullsignal, Osamples) 
            arrays = [qbO]
            cols = ['neg2logL']
            for par, arr in pars[self.analysis.name].items():
                for i in range(arr.shape[-1]):
                    cols += ["{0}_{1}".format(par,i)]
                arrays += [tf.squeeze(arr,axis=1)] # remove 'signal' dimension 
            allpars = tf.concat(arrays,axis=-1)               
            data = pd.DataFrame(allpars.numpy(),columns=cols)
            data.index.rename('EventID',inplace=True) 
            conn = self.connect_to_db()
            c = conn.cursor()
            col_info = [('EventID','integer primary key')] + [(col,'real') for col in cols]
            sql.create_table(c,self.local_table+name+"_observed",col_info)
            sql.upsert(c,self.local_table+name+"_observed",data,primary='EventID')
            self.close_db(conn)

        else:
            # If this isn't the null case, then should also compute the various Asimov likelihoods for this signal, for asymptotic results
            joint0s = JointDistribution(self.analyses,com.deep_merge(signal,self.nullnuis))
            joint0  = JointDistribution(self.analyses,com.deep_merge(self.nullsignal,self.nullnuis))
 
            # Get Asimov samples for nuisance MLEs with fixed signal hypotheses
            samplesAsb = joint0s.Asamples 
            samplesAb = joint0.Asamples

            # Evaluate distributions for Asimov datasets, in case where we
            # know that the MLEs for those samples are the true parameters
            # (i.e. don't have to fit because we just recover the parameters that we put in)
            qsbAsb = -2*(joint0s.log_prob(samplesAsb))
            qbAb  = -2*(joint0.log_prob(samplesAb))
            
            # Fit distributions for Asimov datasets for the other half of each
            # likelihood ratio
            qbAsb, joint_fitted, pars = joint.fit_nuisance(self.nullsignal, samplesAsb,log_tag='bAsb')
            qsbAb, joint_fitted, pars = joint.fit_nuisance(signal, samplesAb,log_tag='sbAsb')

            qAsb = (qsbAsb - qbAsb)[0] # extract single sample result
            qAb = (qsbAb - qbAb)[0]
 
            # ...and fit the observed data too!
            Osamples = joint.Osamples
            qbO , joint_fitted, pars = joint.fit_nuisance(self.nullsignal, Osamples)
            qsbO, joint_fitted, pars = joint.fit_nuisance(signal, Osamples)
            qO = (qsbO - qbO)[0] # extract single sample result

            # Record everything!
            cols = ['qAsb','qAb','qO','neg2logL_quad','neg2logL']
            d = np.array([v.numpy() for v in [qAsb,qAb,qO,qsbO]])
            data = pd.DataFrame(d.T,columns=cols[:-1]) # don't have the full MC likelihood results here, but want the column to exist for future insertion
            data.index.rename('EventID',inplace=True)
            #print("data:",data)
            conn = self.connect_to_db()
            c = conn.cursor()
            col_info = [('EventID','integer primary key')]+[(col,'real') for col in cols]
            sql.create_table(c,self.local_table+name+'_observed',col_info)
            sql.upsert(c,self.local_table+name+'_observed',data,primary='EventID')
            self.close_db(conn)


def collider_analyses_from_long_YAML(yamlfile,replace_SR_names=False):
    """Read long format YAML file of analysis data into BinnedAnalysis objects"""
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
        a = BinnedAnalysis(k,srs,cov,cov_order,ucz)
        analyses[a.name] = a
    if replace_SR_names:
        return analyses, SR_name_translation, inverse_translation
    else:
        return analyses