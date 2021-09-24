Shape handling in jmctf
=======================

Ok, shapes in tensorflow_probability are very confusing, and we need to understand
them thoroughly in order to make jmctf conform to the same conventions, which will
make using tensorflow_probability internally more straightforward.

The following is how tensorflow_probability usually works:
    (sample_shape,batch_shape,event_shape)
When creating a distribution, each independent part will have
    (batch_shape,event_shape)
where event_shape is the shape of one draw from the distribution, for one set of parameters,
and batch_shape is any extra dimensions arising from having tensors of input parameters, i.e.
"batches" of distributions.
sample_shape is created when drawing samples, i.e. if we do distribution.sample(tensor) then
the sample returned will have shape (sample_shape,batch_shape,event_shape), where 
sample_shape = tensor.shape (i.e. if tensor is (20,5) then you generate a 20x5 array of
samples, each with shape (batch_shape,event_shape)).

When computing log_prob, the broadcasting works as follows. Say we do:
   logp = distribution.log_prob(sample)
If sample.shape = (sample_shape,batch_shape,event_shape), where batch_shape and event_shape
match those of 'distribution', then broadcasting is simple: any dimensions of size 1 get
broadcast, and any incompatible dimensions cause an error.

The difficulty is when batch_shape and event_shape don't match between the sample and the distribution. 
Then what happens is the following:
  1. sample.shape is compared to (batch_shape,event_shape). Let len(batch_shape,event_shape)=n. 
     If sample.shape has fewer dimensions, pad them on the left with singleton dimensions.
  2. Broadcast the n right-most dimensions of sample against (batch_shape,event_shape). The remaining
     dimensions are interpreted as sample_shape.
  3. Result will have shape (sample_shape,batch_shape), since each whole event_shape part is used for
     one log_prob evaluation.

Creating distributions of desired event_shapes
----------------------------------------------

We need to also understand how to get a desired (batch_shape,event_shape) combination for a given set
of input parameters for a distribution. This is tricky due to vectorised input parameters. One useful
trick is to use tfp.Independent to transfer dimensions from "batch_shape" into "event_shape". For example,
if we just have one tfd.Normal distribution, we can create an batch_shape=(N,M), event_shape=() by supplying
mu.shape = (N,M). But if we would rather consider this as M Normal distributions, we can do

tdp.Independent(...,reinterpret_batch_dims=1) 

to transfer the dim[1] batch dimension into the event_shape.

 
