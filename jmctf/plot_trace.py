"""Trying to read tensorboard event files"""

import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

x = EventAccumulator(path="massminimize_logs/gof_obs")
x.Reload()
print(x.Tags())

w_times, l_step_nums, l_vals = zip(*x.Tensors('loss[0]'))
#w_times, l_step_nums, l_vals = zip(*x.Tensors('total_loss'))
#w_times, t_step_nums, t_vals = zip(*x.Tensors('theta:0[0]'))
w_times, t_step_nums, t_vals = zip(*x.Tensors('s:0[0]'))

def decode(val):
    tensor_bytes = val.tensor_content
    tensor_dtype = val.dtype
    tensor_shape = [x.size for x in val.tensor_shape.dim]
    tensor_array = tf.io.decode_raw(tensor_bytes, tensor_dtype)
    tensor_array = tf.reshape(tensor_array, tensor_shape)
    return tensor_array

order = np.argsort(l_step_nums)

print(l_step_nums)
l = np.array([decode(l_vals[i]).numpy() for i in order])
print(t_step_nums)
t = np.array([decode(t_vals[i]).numpy() for i in order])

print("t:",t)
print("l:",l)


fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(t,l,s=2)
ax.quiver(t[:-1], l[:-1], t[1:]-t[:-1], l[1:]-l[:-1], angles="xy", scale_units='xy', scale = 1, width=0.002, headwidth=5)
plt.show()

