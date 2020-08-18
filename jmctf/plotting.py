"""Routines to help plot useful things in JMCTF"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow_probability import distributions as tfd

def plot_sample_dist(samples,ax_dict=None,**kwargs):
    # Restructure sample dictionary so it is split into analyses
    out = {}
    for name,val in samples.items():
        aname, vname = name.split("::")
        if aname not in out.keys(): out[aname] = {} 
        out[aname][vname] = val
    return plot_collection(out,ax_dict,**kwargs)

def plot_MLE_dist(MLEs,ax_dict=None,**kwargs):
    return plot_collection(MLEs,ax_dict,**kwargs)

def plot_collection(data,ax_dict=None,**kwargs):
    sizes = {}
    ax_dict_out = {}
    for aname, vdict in data.items():
        sizes[aname] = 0
        for vname,val in vdict.items():
            if len(val.shape) > 2:
                sizes[aname] += val.shape[2]
            else:
                sizes[aname] += 1 
    maxsize = max(sizes.values())
    na = len(data)
    if ax_dict is None:
        fig = plt.figure(figsize=(maxsize*3,na*3))
    for i,(aname,vdict) in enumerate(data.items()):
        j=0
        ax_dict_out[aname] = {}
        for v,val in vdict.items():
            if len(val.shape) == 0: s = 0 # This is a constant! Don't try to plot a distribution for it.
            elif len(val.shape) <= 2: s = 1
            else: s = val.shape[-1]
            ax_dict_out[aname][v] = []
            for k in range(s):
                if ax_dict is None:
                    ax = fig.add_subplot(na,maxsize,j+i*maxsize+1)
                    ax_dict_out[aname][v] += [ax]
                    if j==0: ax.set_title(aname,fontsize=14)
                    if s==1: ax.set_xlabel("{0}".format(v))
                    else:    ax.set_xlabel("{0}[{1}]".format(v,k))
                    ax.set(yscale="log")
                else:
                    ax = ax_dict[aname][v][k]
                if s==1:
                    sns.distplot(val, kde=False, ax=ax, norm_hist=True, **kwargs)
                else:
                    sns.distplot(val[...,k], kde=False, ax=ax, norm_hist=True, **kwargs)
                j+=1
    if ax_dict is None:
        return fig, ax_dict_out
    else:
        return

def plot_chi2(ax,LLR,DOF,yscale="log",c='b'):
    ax.set_xlabel("LLR")
    ax.set(yscale=yscale)
    sns.distplot(LLR, color=c, kde=False, ax=ax, norm_hist=True, label="JMCTF")
    q = np.linspace(0, np.max(LLR),1000)
    chi2 = tf.math.exp(tfd.Chi2(df=DOF).log_prob(q))
    ax.plot(q,chi2,color=c,lw=2,label="chi^2 (DOF={0})".format(DOF))
