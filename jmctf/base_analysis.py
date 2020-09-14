"""Base class for defining independent 'analyses' or 'experiments', ultimately
   to be combined with other such analyses/experiments"""

import tensorflow as tf
import jmctf.common as c

class BaseAnalysis:

    def __init__(self,name):
        self.name = name

    def event_shapes(self):
        """Get a dictionary describing the "event shapes" of data samples for this analysis.
           Basically just the keys of the sample dictionaries plus dimension of each entry
        """
        data = self.get_observed_samples() # Might as well infer it from this data
        structure = {key: tf.constant(val).shape for key,val in data.items()}
        return structure

    def parameter_shapes(self):
        """Get a dictionary describing the primitive (i.e. non batch) shapes of input
           parameters for the analysis"""
        out = {**self.interest_parameter_shapes(),**self.fixed_parameter_shapes(),**self.nuisance_parameter_shapes()}
        return out

    def bcast_parameters_samples(self,parameters,samples):
        """Broadcast dictionaries of distribution parameters against samples
           from those distribtuions, following tensorflow_probability rules
           for log_prob evaluation. See e.g.
           https://www.tensorflow.org/probability/examples/Understanding_TensorFlow_Distributions_Shapes#computing_log_prob_for_scalar_distributions

           Returns dictionaries unchanged except that objects have had shapes changed
           according to the broadcasting rules.

           Does what log_prob functions do internally. Needed for e.g. manually computing MLE initial guesses.
           For reference, here are the rules (see shapes_readme.md):

           When computing log_prob, the broadcasting works as follows. Say we do:
              logp = distribution.log_prob(sample)
           If sample.shape = (sample_shape,batch_shape,event_shape), where batch_shape and event_shape
           match those of 'distribution', then broadcasting is simple: any dimensions of size 1 get
           broadcast, and any incompatible dimensions cause an error.
           
           The difficulty is when batch_shape and event_shape don't match. Then what happens is the following:
             1. sample.shape is compared to (batch_shape,event_shape). Let len(batch_shape,event_shape)=n. 
                If sample.shape has fewer dimensions, pad them on the left with singleton dimensions.
             2. Broadcast the n right-most dimensions of sample against (batch_shape,event_shape). The remaining
                dimensions are interpreted as sample_shape.
             3. Result will have shape (sample_shape,batch_shape), since each whole event_shape part is used for
                one log_prob evaluation.

           TODO: actually I don't think I am quite following the rules properly, not doing any padding here. Just assuming that
           the batch_shapes can be broadcast against each others.
        """

        # Infer "distribution" batch shape from parameters.
        dist_batch_shape = c.dist_batch_shape(parameters,self.parameter_shapes())

        # Infer sample+batch_shape from samples
        sample_batch_shape = c.sample_batch_shape(samples,self.event_shapes())
              
        #print("samples",samples)
        #print("sample_batch_shape:",sample_batch_shape)
        #print("parameters:",parameters)
        #print("dist_batch_shape:",dist_batch_shape)

        try:
            # Attempt to broadcast these shapes together
            final_batch_shape = c.get_bcast_shape(dist_batch_shape,sample_batch_shape)
        except ValueError as e:
            msg = "Failed to broadcast batch shapes for samples ({0}) against the batch shape for the distribution ({1}, inferred from input parameter shapes). The broadcasting rules for these follow the 'log_prob' evaluation rules defined by tensorflow_probabilty, please ensure your input parameters and samples are broadcastable by those rules.".format(sample_batch_shape,dist_batch_shape)
            raise ValueError(msg) from e

        # Broadcast both the parameters and samples to this inferred batch shape
        out_samples = c.bcast_sample_batch_shape(samples,self.event_shapes(),final_batch_shape)
        out_pars    = c.bcast_dist_batch_shape(parameters,self.parameter_shapes(),final_batch_shape)

        return out_pars, out_samples
