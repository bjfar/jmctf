"""Assorted common helper functions"""

import yaml
import sqlite3
import numpy as np
import tensorflow as tf
import scipy.interpolate as spi
from collections.abc import Mapping

# Reference dtype for consistency in TensorFlow operations
TFdtype = np.float32

# Smallest positive 32-bit float
nearlyzero32 = np.nextafter(0,1,dtype=TFdtype)

# A constant "close" to nearlyzero32, but large enough to avoid divide-by-zero
# nans, and similar floating point problems, in all parts of JMCTF
# (but still small enough to avoid observable errors in results)
# TODO: Make sure this is sufficiently robust
reallysmall = 1e10*nearlyzero32

# Stuff to help format YAML output
class blockseqtrue( list ): pass
def blockseqtrue_rep(dumper, data):
        return dumper.represent_sequence( u'tag:yaml.org,2002:seq', data, flow_style=True )
yaml.add_representer(blockseqtrue, blockseqtrue_rep)

# Helpers for storing numpy arrays in sqlite tables
# (see https://stackoverflow.com/a/55799782/1447953)
compressor = 'zlib'  # zlib, bz2

def to_numpy(d):
    """Convert bottom level items in nested dictionary
       of TensorFlow tensors into numpy arrays"""
    out = {}
    for k,v in d.items():
        if isinstance(v, dict): out[k] = to_numpy(v)
        else: out[k] = v.numpy()
        return out

def deep(d_arg=0):
    def deep_decorator(f):
        """Decorator to make a function apply over the
           bottom level of nested dictionaries.
           d_arg specified which argument of the
           function is to be treated as the dictionary"""
        def deep_f(*args,**kwargs):
            if isinstance(args[d_arg], Mapping):
                out = {}
                for k,v in args[d_arg].items():
                    args_list = list(args)
                    args_list[d_arg] = v
                    out[k] = deep_f(*args_list,**kwargs)
            else:
                out = f(*args,**kwargs)
            return out
        return deep_f
    return deep_decorator

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
    cdf = tf.constant(np.arange(1, x.shape[0]+1)/float(x.shape[0]),dtype=TFdtype)
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
    """From https://stackoverflow.com/a/56177639/1447953

    Merge two values, with `b` taking precedence over `a`.
     
    Semantics:

     * If either `a` or `b` is not a dictionary, `a` will be returned only if
       `b` is `None`. Otherwise `b` will be returned.

     * If both values are dictionaries, they are merged as follows:
     
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

def print_with_id(d,id_only=False):
    """Print dictionary contents along with their memory addresses in hex format,
       to help identify which contents refer to the same objects"""
    if d is not None:
      out = "{"
      for k,v in d.items():
        print(v)
        out += "{0}: ".format(k)
        if isinstance(v, Mapping):
            out+="{0}".format(print_with_id(v,id_only))
        elif id_only:
            out+="address={0}, ".format(id(v))
        else:
            out+="{0} (address={1}), ".format(v,id(v))
      out += "}"
    else:
        out = "None"
    return out

def convert_to_TF_variables(d):
   """Convert bottom-level entries of dictionary to TensorFlow 'Variable' objects, to be fed to
      TensorFlow optimiser and gradient calculations"""
   out = {}
   #print("constant=",constant)
   for k,val in d.items():
       if isinstance(val,tf.Variable):
           out[k] = val # Already a Variable, no need to convert
           #print(k, 'was tf.Variable, left alone:', val)
       elif isinstance(val, Mapping):
           out[k] = convert_to_TF_variables(val) # We must go deeper
       else:
           # Try to create tensorflow variable from this data
           try:
               out[k] = tf.Variable(val, dtype=TFdtype)
               #print(k, 'converted to Variable:', val)
           except Exception as e:
               msg = "Failed to convert values for key {0} to a TensorFlow Variable object! See associated exception for more information. Values were: {1}".format(k,val)
               raise ValueError(msg) from e
   return out

def convert_to_TF_constants(val,ignore_variables=False):
   """Convert bottom-level entries of dictionary to TensorFlow 'Tensor' objects
      If ignore_variables is True then dictionary values that are TensorFlow
      'Variable' objects will be left unchanged. Otherwise, their existence in
      the input dictionary will cause an error to be raised.
   """
   if isinstance(val,tf.Variable):
       if ignore_variables:
           # No conversion
           out = val
       else:
           # Shouldn't be trying to convert Variables to constant tensors!
           msg = "Attempted to convert dictionary entry with key '{0}' (which is a TensorFlow 'Variable' object) into a constant TensorFlow 'Tensor' object! This conversion is not allowed, please make sure that, for example, parameters returned from Analysis objects are not already Variable objects."
           raise TypeError(msg)
   elif isinstance(val,tf.Tensor):
       out = val
       #print(k, 'was already tf.Tensor, left alone:', val)  
   elif isinstance(val, Mapping):
       out = {}
       for k,v in val.items():
           out[k] = convert_to_TF_constants(v,ignore_variables) # We must go deeper
   else:
       # Try to create tensorflow variable or constant from this data
       try:
           out = tf.constant(val, dtype=TFdtype)
           #print(k, 'converted to tf.Tensor:', val)
       except Exception as e:
           msg = "Failed to convert values for key {0} to a TensorFlow Tensor object! See associated exception for more information. Values were: {1}".format(k,val)
           raise ValueError(msg) from e
   return out

# Nested list/tuple flattener. From https://stackoverflow.com/a/10824420/1447953
# Modified to also suck values out of dicts (discarding keys)
def flatten(container):
    if isinstance(container, (Mapping)):
        for i in flatten(list(container.values())):
            yield i
    else:
        for i in container:
            if isinstance(i, (list,tuple)):
                for j in flatten(i):
                    yield j
            elif isinstance(i, (Mapping)):
                for j in flatten(list(i.values())):
                    yield j
            else:
                yield i

def atleast_2d(pars,report=False,new1Daxis=1):
    """Ensure parameters in nested dictionaries meet the shape conventions of JointDistribution, i.e.
       all pars must be 2D (as expanded by atleast_2d)
       Doesn't work quite like the numpy version by default: new dimensions
       for 1D input are added at axis 1 rather than 0. But setting new1Daxis=0 gives numpy behaviour.
       If 'report' is True, also return a flag indicating whether any dimension expansion occured. 
    """
    did_expansion = False
    if isinstance(pars, Mapping):
        out = {}
        for k,v in pars.items():
            out[k], r = atleast_2d(v,True)
            if r: did_expansion = True
    elif pars.shape==():
        # Corner case for scalar tensors
        out = tf.expand_dims(tf.expand_dims(pars,axis=0),axis=0)
        did_expansion = True
    elif len(pars.shape)==1:
        out = tf.expand_dims(pars,axis=new1Daxis)
        did_expansion = True               
    else:
        out = pars
        # TODO: error for not tensor

    if report:
        return out, did_expansion
    else:
        return out

def extract_ith(d,i,keep_axis=False):
    """Extract ith entry from bottom level of all nested dictionaries.
       If keep_axis=True then the axis from which the entry is extracted
       is kept as a singleton dimension."""
    if isinstance(d,Mapping):
        out = {}
        for k,v in d.items():
            out[k] = extract_ith(v,i,keep_axis)
    elif d.shape==():
        if i==0:
            # We will allow extraction of a scalar using index 0
            out = d
        else:
            msg = "Tried to extract from a scalar using index other than zero! Index was {0}, bottom-level item was {1}".format(i,d)
            raise IndexError(msg)
    elif keep_axis:
        out = d[i:i+1]
    else:
        out = d[i]
    return out

def deep_size(d,axis=0):
    """Measure the size along an axis of all bottom level objects in nested dictionaries.
       If size is consistent, return it, if dict(s) is empty returns -1, otherwise returns None"""
    if isinstance(d, Mapping):
        size = -1
        for k,v in d.items():
            size_v = deep_size(v,axis)
            if size==-1 or size==1: # size 1 can be broadcast to larger sizes, so this is ok to be different 
                size = size_v
            elif size_v==1: # Again, can be broadcast to whatever 'size' is.
                pass
            elif size!=size_v:
                size = None
    else:
        size = d.shape[axis]
    return size

def deep_shape(d,axis=0,numpy_bcast=True):
    """Measure the size along an axis of all bottom level objects in nested dictionaries.
       If consistent (in broadcast terms) shape can be found, return it, otherwise an error is thrown."""
    if isinstance(d, Mapping):
        shape = None
        for k,v in d.items():
            shape_v = deep_shape(v,axis)
            if shape == None:
                shape = shape_v
            shape = get_bcast_shape(shape,shape_v,numpy_bcast)
    else:
        shape = d.shape
    return shape

def get_bcast_shape(shape1,shape2,numpy_bcast=True):
    """Of two tensor/array shapes, return the shape to which they can be broadcast under numpy rules
       (or throw an error if bcast is not possible"""
    print("shape1:", shape1, len(shape1))
    print("shape2:", shape2, len(shape2))
    if len(shape1)!=len(shape2):
        if numpy_bcast:
            # Apply numpy broadcasting rules to match dimensions and add "implicit" dimensions
            if len(shape1)<len(shape2):
                # Expand shape1 with ones on the left
                newshape1 = [1 for s in shape2]
                if len(shape1)!=0: newshape1[-len(shape1)::] = shape1
                shape1 = tuple(newshape1)
            else:
                # Expand shape2 with ones on the left
                newshape2 = [1 for s in shape1]
                if len(shape2)!=0: newshape2[-len(shape2)::] = shape2
                shape2 = tuple(newshape2) 
        else:
            # Without full numpy broadcasting, shape is indeterminate 
            msg = "Shapes are not compatible! They have different numbers of dimensions (found {0} vs {1})".format(shape1,shape2)
            raise ValueError(msg)
    print("expanded shape1:", shape1, len(shape1))
    print("expanded shape2:", shape2, len(shape2))
    # Number of dimensions should now match. Now compare their sizes.
    newshape = [-1 for s in shape1]
    for i,(si,sv) in enumerate(zip(shape1,shape2)):
        if si==sv:
            newshape[i] = si
        elif si==1:
            newshape[i] = sv
        elif sv==1:
            newshape[i] = si
        else:
            msg = "Shapes are not compatible! Different sized dimensions (other than size 1) were found! (found {0} vs {1}, where (at least) dim {3} are not compatible".format(shape1,shape2,i)
            raise ValueError(msg)
    if -1 in newshape:
        msg = "Failed to broadcast shapes! -1 detected! (shape1 = {0}, shape2={1}, newshape = {2})".format(shape1,shape2,newshape)
        raise ValueError(msg)
    print("newshape: {0}".format(newshape))
    return tuple(newshape)

def squeeze_axis_0(pars):
    """Examines parameter (i.e. nested) dictionary and, if it can be interpreted
       as describing just a single hypothesis, squeeze out the
       hypothesis dimension (i.e axis 0).
       Also ensures that dim 0 is consistent size for all bottom-level objects"""
    # Ensure that 2d form is initially used
    pars_2d = atleast_2d(pars)
    print("pars:", pars)
    print("pars_2d:",pars_2d)
    size_axis0 = deep_size(pars,axis=0)
    print("size_axis0:", size_axis0)
    if size_axis0==1:
        out = deep_squeeze(pars,axis=0)
    elif size_axis0==None:
        msg = "Error while squeezing axis 0 for objects in nested dictionary! Objects did not have consistent size of axis 0"
        raise ValueError(msg) 
    else:
        out = pars # Do nothing to parameters, not even the shape adjustment
    return out

def squeeze_to(tensor,d,dont_squeeze=[]):
    """Squeeze a tensor down to d dimensions.
       Throws error if this is not possible.
       Just squeezes singleton dimensions starting
       from the left until dimensions matches d.
       Dimensions in dont_squeeze list will not
       be squeezed.
    """
    squeezeable_dims = [d for d in tf.where(tf.equal(tensor.shape,1))[:,0].numpy() if d not in dont_squeeze]
    nextra = len(tensor.shape) - d
    if nextra<0:
        msg = "Could not squeeze tensor to {0} dimensions! Starting dimension ({1}) is larger than target dimension!".format(d,len(tensor.shape))
        raise ValueError(msg)
    elif nextra==0:
        out = tensor # Already correct dimension
    elif len(squeezeable_dims)<nextra:
        msg = "Could not squeeze {0}d tensor to {1} dimensions! Not enough squeezeable dimensions! (len(squeezeable_dims) = {2}, tensor.shape = {3}, dont_squeeze = {4})".format(len(tensor.shape), d, len(squeezeable_dims), tensor.shape, dont_squeeze)
        raise ValueError(msg)
    else:
        out = tf.squeeze(tensor,axis=squeezeable_dims[:nextra])
    return out

@deep(1)
def deep_einsum(instructions,d):
    """Perform a tf.einsum on the bottom level of nested dictionaries"""
    return tf.einsum(instructions,d)

@deep(0)
def deep_squeeze(pars,axis):
    """Apply tf.squeeze to all bottom-level objects in nested dictionaries"""
    return tf.squeeze(pars,axis=axis) # Should be pre-checked to be size 1

@deep(0)
def deep_expand_dims(d,axis):
    """Apply tf.expand_dims to all bottom-level objects in nested dictionaries"""
    return tf.expand_dims(d,axis=axis)

@deep(0)
def deep_broadcast(d,shape):
    """Apply tf.broadcast to all bottom-level objects in nested dictionaries"""
    return tf.broadcast_to(d,shape)

def deep_equals(d,val=None):
    """Return True (and the value) if all bottom-level objects in nested dictionaries are equal"""
    if isinstance(d, Mapping):
        out_val = val
        out_equal = True
        for k,v in d.items():
            deep_equal, deep_val = deep_equals(v,out_val)
            out_val = deep_val
            out_equal = out_equal and deep_equal
            if not out_equal:
                break # Short-circuit for non-equal outcome
    else:
        if val is None:
            out_equal = True
            out_val = d
        elif tf.reduce_all(tf.equal(d,val)):
            out_equal = True
            out_val = val
        else:
            out_equal = False
            out_val = None
    return out_equal, out_val

def sample_batch_shape(sample,event_shapes):
    """Work out the effective 'sample shape + batch shape' for a sample, given a dictionary of
       event shapes.
       Error if consistent batch dims are not found, or if any required event shapes
       are missing.
    """
    batch_shape = None
    for name,x in sample.items():
        eshape = event_shapes[name]
        if len(eshape)>0 and x.shape[-len(eshape):] != eshape:
            msg = "Supplied event shape for distribution {0} ({1}) does not match trailing dimensions of sample! ({2})!".format(name,eshape,x.shape)
            raise ValueError(msg)
        bshape = x.shape[:-len(eshape)]
        if batch_shape is None:
            batch_shape = bshape
        elif batch_shape!=bshape:
            msg = "Inferred batch shapes for samples are not consistent! Inferred batch shape of {0} for samples from distribution {1}, but previously found batch shapes of {2}".format(bshape,name,batch_shape)
            raise ValueError(msg)
    return batch_shape

def loose_squeeze(tensor,axis):
    """Applies squeeze to bottom-level dict objects along axis, but isn't an error if the axis
       is not size 1 (just does nothing in that case)"""
    if isinstance(tensor, Mapping):
        out = {}
        for k,v in tensor.items():
            out[k] = loose_squeeze(v,axis)
    elif tensor.shape[axis]==1: 
        out = tf.squeeze(tensor,axis=axis)
    else:
        out = tensor
    return out

def cat_pars_2d(pars,remove_axis0=False):
    """Stack separate tensorflow parameters from a dictionary into
       a single tensor, in order fixed by the dict default iteration order.
       Ensures that output is at least 2D, i.e. scalar input has two dimensions
       added, and 1D input has one dimension added (as axis 0).
       If remove_axis0 is True, then AFTER expanding everything to 2D, axis 0
       is squeezed. It will be an error to use this option for input for which
       the (expanded) axis 0 is not a singleton.

       TODO: Allow 2D parameters? Could automatically flatten/unflatten them to 1D.
    """
    parlist = []
    maxdims = {}
    for ka,a in atleast_2d(pars).items():
        for kp,p in a.items():
            if remove_axis0:
                if p.shape[0]!=1:
                    msg = "Tried to squeeze axis 0 due to 'remove_axis0=True' flag, but this axis was not a singleton for parameter {0} of analysis {1}. This parameter had original shape {2} and expanded shape {3}".format(kp,ka,p.shape,p.shape)
                    raise ValueError(msg)
                else:
                    p = tf.squeeze(p,axis=0)
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
    bcast = tf.broadcast_to(tf.constant(np.ones([1 for d in range(len(maxdims))]),dtype=TFdtype),maxshape)
    for p in parlist:
        matched_parlist += [p*bcast]
    #return tf.Variable(tf.concat(matched_parlist,axis=-1),name="all_parameters")               
    return tf.concat(matched_parlist,axis=-1) 

def uncat_pars_2d(catted_pars,pars_template,axis0_removed=False):
    """De-stack tensorflow parameters back into separate variables of
       shapes know to each analysis. Assumes stacked_pars are of the
       same structure as pars_template. This version reverses
       cat_pars_2d, i.e. it removes singleton dimensions as found to
       match scalar and 1D entries in pars_template.
       If axis0_removed is True, it is assumed that the catted_pars
       have had their 0th axis squeezed, and this is restored during
       the de-catting"""
    pars = {}
    i = 0
    if axis0_removed:
        # Restore singleton axis 0 if it was removed
        catted_pars_2d = tf.expand_dims(catted_pars,axis=0)              
    else:
        catted_pars_2d = catted_pars

    for ka,a in pars_template.items():
        pars[ka] = {}
        for kp,p in a.items():
            # We always stack to at least 2D, so need to undo this for some cases
            if p.shape==():
                N = 1 # Even scalars still take up one slot
                piN = tf.squeeze(catted_pars_2d[...,i:i+N],axis=[0,1])
            elif len(p.shape)==1:
                N = 1 # Assume there is an implicit singleton inner dimension
                piN = catted_pars_2d[...,i:i+N]        
            else:
                # No change to shape required
                N = p.shape[-1]
                piN = catted_pars_2d[...,i:i+N]
            pars[ka][kp] = piN
            i+=N
    return pars

def iterate_samples(sample_dict):
    """Iterate through samples in a dictionary of samples"""
    i = 0
    N = list(sample_dict.values())[0].shape[0]
    while i<N:
        out = {}
        for name,x in sample_dict.items():
            out[name] = x[i]
        yield out
        i+=1
