"""Unit tests for LEECorrectorMaster and related classes"""

import pytest
import numpy as np
from pathlib import Path
import shutil
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import jmctf.common as c
from jmctf import JointDistribution
from jmctf_tests.analysis_class_register import get_id_list, get_obj, get_test_hypothesis, get_hypothesis_lists, analysis_tests
from jmctf.lee import LEECorrectorMaster

N = 10 # Number of samples to generate in tests

# Hypothesis generator function for use with LEE in tests
# We actually need to provide a function that *creates* the generator since we need to run it multiple times.
def get_hyp_gen(hypotheses):
    def hyp_gen():
        chunk_size = 10
        j = 0
        while True:
            chunk = {name: {par: dat[j:j+chunk_size] for par,dat in a.items()} for name,a in hypotheses.items()}
            this_chunk_size = c.deep_size(chunk)
            if this_chunk_size==0: # Or some other error?
                return # Finished
            ids = list(range(j,j+this_chunk_size))
            print("chunk {0}:{1} = {2}, ids = {3}, this_chunk_size = {4}".format(j,j+chunk_size,chunk,ids,this_chunk_size))
            j += this_chunk_size
            yield chunk, ids
    return hyp_gen

# Special version for single hypothesis case
def get_hyp_gen_1(hypothesis):
    def hyp_gen():
        hyp_list = c.deep_expand_dims(hypothesis,0)
        yield hyp_list, [0]
        return
    return hyp_gen

@pytest.fixture(scope="module"
               ,params=[(get_obj(name), get_test_hypothesis(name), get_hypothesis_lists(name)) for name in analysis_tests.keys()]
               ,ids = [name for name in analysis_tests.keys()]
                )
def analysis_params(request):
    analysis, params1, paramsN = request.param
    return analysis, params1, paramsN 

@pytest.fixture(scope="module")
def analysis(analysis_params):
    analysis, params1, paramsN = analysis_params
    return analysis 

@pytest.fixture(scope="module")
def params1(analysis_params):
    analysis, params1, paramsN = analysis_params
    pars = {analysis.name: params1}
    return pars

@pytest.fixture(scope="module")
def paramsN(analysis_params):
    analysis, params1, paramsN = analysis_params
    pars = {analysis.name: paramsN}
    return pars

@pytest.fixture(scope="module")
def lee(analysis,params1):
    null_hyp = params1
    path = "unit_test_output"
    master_name = list(params1.keys())[0] + "_lee_unit_test"
    nullname = "params1"
    # Clear out old output from previous tests
    # TODO: Should probably make a method to do this? Though the data can be valuable, can be nice to force manual deletion.
    dirpath = Path(path)
    if dirpath.exists(): 
        if dirpath.is_dir():
            shutil.rmtree(dirpath)
        else:
            msg = "{0} exists on system but isn't a directory! This is preventing the creation of output files necessary for running this unit test."
            raise OSError(msg)
    # Create lee object
    lee = LEECorrectorMaster([analysis],path,master_name,null_hyp,nullname)
    return lee

# Function scopes version for certain special tests
@pytest.fixture(scope="function")
def fresh_lee(analysis,params1):
    null_hyp = params1
    path = "unit_test_output"
    master_name = list(params1.keys())[0] + "_fresh_lee_unit_test"
    nullname = "params1"
    # Clear out old output from previous tests
    # TODO: Should probably make a method to do this? Though the data can be valuable, can be nice to force manual deletion.
    dirpath = Path(path)
    if dirpath.exists(): 
        if dirpath.is_dir():
            shutil.rmtree(dirpath)
        else:
            msg = "{0} exists on system but isn't a directory! This is preventing the creation of output files necessary for running this unit test."
            raise OSError(msg)
    # Create lee object
    lee = LEECorrectorMaster([analysis],path,master_name,null_hyp,nullname)
    return lee

@pytest.fixture(scope="module")
def add_events(lee): 
    lee.add_events(N)

j = 0
@pytest.fixture(scope="module")
def process_null(lee,add_events):
    global j
    print("Running {0}th time: lee.db={1}".format(j,lee.db))
    j+=1
    lee.process_null()

i = 0
@pytest.fixture(scope="module")
def process_alternate(lee,add_events,paramsN):
    global i
    print("Running {0}th time: lee.db={1}".format(i,lee.db))
    i+=1
    lee.process_alternate(get_hyp_gen(paramsN))

def test_construct_lee(lee):
    pass

def test_add_events(add_events):
    pass

def test_process_null(process_null):
    pass

def test_process_alternate(process_alternate):
    pass

def test_trivial_alternate(fresh_lee,params1):
    """Tests that same log_prob values are obtained for
       null and alternate fits when the only alternate
       hypothesis *is* the null.
       Also checks that results match 'vanilla' results
       from JointDistribution"""
    # Need a largish number of events here to detect some rare problems
    fresh_lee.add_events(1000)
    fresh_lee.process_null()
    fresh_lee.process_alternate(get_hyp_gen_1(params1))
    df_null, df_null_obs = fresh_lee.load_results(fresh_lee.null_table,['log_prob'],get_observed=True)
    df_prof, df_prof_obs = fresh_lee.load_results(fresh_lee.profiled_table,['log_prob_quad','logw'],get_observed=True)
    print("df_null:", df_null)
    print("df_null['log_prob']:",df_null['log_prob'])
    print("df_prof:", df_prof)
    print("df_prof['log_prob_quad']:",df_prof['log_prob_quad'])
    print("df_null_obs:", df_null_obs)
    print("df_prof_obs:", df_prof_obs)
    # Get 'vanilla' JointDistribution results
    samples = fresh_lee.load_all_events() # Loads all events currently on disk
    joint = JointDistribution(fresh_lee.analyses,params1)
    log_prob, joint_fitted, par_dict = joint.fit_nuisance(samples,params1)
    print("log_prob:", log_prob)
    print("df_null['log_prob'] - log_prob:", df_null['log_prob'] - log_prob[:,0,0])
    print("df_prof['log_prob_quad'] - log_prob:", df_prof['log_prob_quad'] - log_prob[:,0,0])
    print("df_null['log_prob'] - df_prof['log_prob_quad']:", df_null['log_prob'] - df_prof['log_prob_quad'])
    tol = 1e-6
    assert ((df_null['log_prob'] - log_prob[:,0,0]).abs() < tol).all() # LEE null vs JointDistribution
    assert ((df_prof['log_prob_quad'] - log_prob[:,0,0]).abs() < tol).all() # LEE (quad) alternate vs JointDistribution
    assert ((df_null['log_prob'] - df_prof['log_prob_quad']).abs() < tol).all() # LEE: null vs (quad) alternate (redundant but why not do it)
    assert ((df_null_obs['log_prob'] - df_prof_obs['log_prob_quad']).abs() < tol).all() # LEE: null vs alternate (obs)
