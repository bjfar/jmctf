"""Base class for defining independent 'analyses' or 'experiments', ultimately
   to be combined with other such analyses/experiments"""

class BaseAnalysis:

    def __init__(self,name):
        self.name = name

    def event_shapes(self):
        """Get a dictionary describing the "event shapes" of data samples for this analysis.
           Basically just the keys of the sample dictionaries plus dimension of each entry
           NOTE: Currently assumes events are at most 1D
           """
        data = self.get_observed_samples() # Might as well infer it from this data
        structure = {key: val.shape[-1] for key,val in data.items()}
        return structure

    def parameter_shapes(self):
        """Get a dictionary describing the primitive (i.e. non batch) shapes of input
           parameters for the analysis"""
        out = {**self.interest_parameter_shapes(),**self.fixed_parameter_shapes(),**self.nuisance_parameter_shapes()}
        return out
