import yaml
import pandas as pd
import tensorflow as tf
from tensorflow_probability import distributions as tfd

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

    def tensorflow_model(self,signal,nuispars):
        """Output tensorflow probability model object, to be combined together and
           sampled from. Signal parameters should all be fixed."""

        # Need to construct these shapes to match the event_shape, batch_shape, sample_shape 
        # semantics of tensorflow_probability.

        cov_order = self.cov_order
        if self.cov is not None:
            if cov_order is None or cov_order=="use SR order": cov_order = self.SR_names 
 
        tfds = []
        for sr, bexp, bsysin in zip(self.SR_names, self.SR_b, self.SR_b_sys):
            s = signal[sr]
            b = tf.constant(bexp,dtype=float)
            bsys = tf.constant(bsysin,dtype=float)

            # Poisson model
            poises0  = tfd.Poisson(rate = s+b)
            tfds += [poises0]

            if self.cov is None or sr not in cov_order:
                # if no covariance info for this SR:
                #  Either 1. combine assuming independent, or 2. need to select one SR to use.
                #  For now just do 1. TODO: will need 2 also at some point
         
                # Nuisance parameters, use nominal null values (independent Gaussians to model background/control measurements)
                theta = nuispars[sr]
                nuis0 = tfd.Normal(loc = theta, scale = bsys) # I forget if bsys includes the Poisson uncertainty on b TODO: Check this!
                tfds += [nuis0]

        if self.cov is not None:
            # Construct multivariate normal model for background/control measurements
            theta_vec = tf.stack([nuispars[sr] for sr in cov_order],axis=0) 
            cov = tf.constant(self.cov,dtype=float)
            cov_nuis = tfd.MultivariateNormalFullCovariance(loc=theta_vec,covariance_matrix=cov)
            tfds += [cov_nuis]

        return tfds

    def tensorflow_null_model(self,signal):
        # Make sure signal consists of constant tensorflow objects
        sig = {sr: tf.constant(signal[sr], dtype=float) for sr in self.SR_names}
        zeros = {sr: tf.constant(0, dtype=float) for sr in self.SR_names}
        tfds = self.tensorflow_model(sig,zeros)
        return tfds

    def get_tensorflow_variables(self):
        """Get parameters to be optimized, for input to "tensorflow_free_model"""
        # TODO: initial value (guess) for nuisance pars needs to be set carefully 
        thetas = {sr: tf.Variable(0, dtype=float, name='theta_{0}'.format(sr)) for sr in self.SR_names}
        return thetas

    def tensorflow_free_model(self,signal,thetas):
        """Output tensorflow input parameter tensors and probability objects for this analysis,
           for optimisation/profiling relative to some simulated data
           Input:
              dictionary of signal parameter tensors to use
        """

        sig = {sr: tf.constant(signal[sr], dtype=float) for sr in self.SR_names}
        tfds = self.tensorflow_model(sig,thetas)
        return tfds 
       
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
