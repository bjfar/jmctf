"""Base class for defining independent 'analyses' or 'experiments', ultimately
   to be combined with other such analyses/experiments"""

class BaseAnalysis:

    def __init__(self,name):
        self.name = name

    def get_sample_structure(self):
        """Get a dictionary describing the structure of data samples for this analysis.
           Basically just the keys of the sample dictionaries plus dimension of each entry"""
        data = self.get_observed_samples() # Might as well infer it from this data
        structure = {key: val.shape[-1] for key,val in data.items()}
        return structure
