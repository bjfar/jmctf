"""Assorted common helper functions"""

import yaml
import sqlite3
import numpy as np
import tensorflow as tf
import scipy.interpolate as spi

# Stuff to help format YAML output
class blockseqtrue( list ): pass
def blockseqtrue_rep(dumper, data):
        return dumper.represent_sequence( u'tag:yaml.org,2002:seq', data, flow_style=True )
yaml.add_representer(blockseqtrue, blockseqtrue_rep)

# Helpers for storing numpy arrays in sqlite tables
# (see https://stackoverflow.com/a/55799782/1447953)
compressor = 'zlib'  # zlib, bz2

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    # zlib uses similar disk size that Matlab v5 .mat files
    # bz2 compress 4 times zlib, but storing process is 20 times slower.
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    # Compression didn't seem to make much difference so I turned it off
    #return sqlite3.Binary(codecs.encode(out.read(),compressor))  # zlib, bz2
    return sqlite3.Binary(out.read())  # zlib, bz2

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    #out = io.BytesIO(codecs.decode(out.read(),compressor))
    return np.load(out)

#def adapt_array(arr):
#    return arr.tobytes()
#def convert_array(text):
#    return np.frombuffer(text)

sqlite3.register_adapter(np.ndarray, adapt_array)    
sqlite3.register_converter("array", convert_array)

def eCDF(x):
    """Get empirical CDFs of arrays of samples. Assumes first dimension
       is the sample dimension. All CDFs are the same since number of
       samples has to be the same"""
    cdf = tf.constant(np.arange(1, x.shape[0]+1)/float(x.shape[0]),dtype=float)
    #print("cdf.shape:", cdf.shape)
    #print("x.shape:", x.shape)
    return cdf
    #return tf.broadcast_to(cdf,x.shape)

def CDFf(samples,reverse=False,return_argsort=False):
    """Return interpolating function for CDF of some simulated samples"""
    if reverse:
        s = np.argsort(samples[np.isfinite(samples)],axis=0)[::-1] 
    else:
        s = np.argsort(samples[np.isfinite(samples)],axis=0)
    ecdf = eCDF(samples[s])
    CDF = spi.interp1d([-1e99]+list(samples[s])+[1e99],[ecdf[0]]+list(ecdf)+[ecdf[1]])
    if return_argsort:
        return CDF, s #pvalue may be 1 - CDF(obs), depending on definition/ordering
    else:
        return CDF

def gather_by_idx(x,indices):
    idx = tf.cast(indices,dtype=tf.int32)
    idx_flattened = tf.range(0, x.shape[0]) * x.shape[1] + idx
    y = tf.gather(tf.reshape(x, [-1]),  # flatten input
                  idx_flattened)  # use flattened indices
    return y

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

def add_prefix(prefix,d):
    """Append the supplied string to all keys in the supplied dictionary.
       Used to add unique prefixes to sample/parameter names to avoid
       clashes when analyses are combined into one big tensorflow_probability
       model"""
    return {"{0}::{1}".format(prefix,key): val for key,val in d.items()}
   
def remove_prefix(prefix,d):
    """Inverse of add_prefix"""
    return {key.split("{0}::".format(prefix))[1]: val for key,val in d.items()}

