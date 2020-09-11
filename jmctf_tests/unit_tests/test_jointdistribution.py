"""Unit tests for JointDistribution class"""

import pytest
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import jmctf.common as c
from jmctf import JointDistribution
from jmctf_tests.analysis_class_register import get_id_list, get_obj, get_test_hypothesis, get_hypothesis_lists

# Common jmctf_test fixtures needed by these tests
from jmctf_tests.common_fixtures import analysis, pars, pars0, joint0, samples, joint_fitted_nuisance_log_prob_pars, joint_fitted_nuisance, fitted_log_prob, fitted_pars, hessian, internal_pars,  decomposed_parameters, decomposed_hessian, quad_prep, quad_f

# Number of samples to draw during tests
N = 1000
@pytest.fixture(scope="module")
def Nsamples():
    return N

# Hessian test shape outputs
Hessians_shape = {
        "NormalAnalysis": (N, 1, 1) ,
        "NormalTEAnalysis": (N, 2, 2) ,
        "BinnedAnalysis": (N, 4, 4)
        }

# @pytest.fixture(scope="module"
#                ,params=[(get_obj(name), get_test_hypothesis(name), H) for name,H in Hessians_shape.items()]
#                       +[(get_obj(name), get_hypothesis_lists(name), H) for name,H in Hessians_shape.items()]
#                ,ids = [name + " (single hypothesis)" for name in Hessians_shape.keys()]
#                      +[name + " (multiple hypotheses)" for name in Hessians_shape.keys()]
#                 )

# "Entry point" fixture providing the analysis objects and parameters for them.
@pytest.fixture(scope="module"
               ,params=[(get_obj(name), get_hypothesis_lists(name), H) for name,H in Hessians_shape.items()]
               ,ids = [name + " (multiple hypotheses)" for name in Hessians_shape.keys()]
                )
def analysis_params_Hshape(request):
    analysis, params, H_shape = request.param
    return analysis, params, H_shape 

@pytest.fixture(scope="module")
def analysis_params(analysis_params_Hshape):
    analysis, params, H_shape = analysis_params_Hshape
    return analysis, params

@pytest.fixture(scope="module")
def samples(joint0,Nsamples):
    x = joint0.sample(Nsamples)
    print("x:", x) 
    return x 

@pytest.fixture(scope="module")
def Hshape(analysis_params_Hshape):
    analysis, params, H_shape = analysis_params_Hshape
    return H_shape 

@pytest.fixture(scope="module")
def sub_hessian_shapes(hessian,decomposed_parameters,decomposed_hessian,Hshape):
    H, g = hessian
    Hii, Hnn, Hin = decomposed_hessian
    interest_i, interest_p, nuisance_i, nuisance_p = decomposed_parameters
    # Compute number of slots in Hessian that each parameter should occupy
    # Assumes that parameters were correctly decomposed (should have a separate test for this)
    islots = 0
    for pars in interest_i.values():
        for (i,n) in pars.values():
            islots += n
    nslots = 0
    for pars in nuisance_i.values():
        for (i,n) in pars.values():
            nslots += n
    if islots>0 and nslots>0:
        Hii_shape = (Hshape[0],islots,islots)
        Hnn_shape = (Hshape[0],nslots,nslots)
        Hin_shape = (Hshape[0],islots,nslots)
    elif nslots>0:
        Hii_shape = None
        Hnn_shape = H.shape #(Hshape[0],nslots,nslots)
        Hin_shape = None
    elif islots>0:
        Hii_shape = H.shape #(Hshape[0],islots,islots)
        Hnn_shape = None
        Hin_shape = None
    else:
        # No parameters of any kind?
        Hii_shape = None
        Hnn_shape = None
        Hin_shape = None
    return Hii_shape, Hnn_shape, Hin_shape
  
def test_fitted_log_prob(joint_fitted_nuisance,fitted_log_prob,samples):
    recalculated_log_prob = joint_fitted_nuisance.log_prob(samples)
    print("fitted_log_prob:", fitted_log_prob)
    print("recalculated_log_prob:", recalculated_log_prob)
    print("joint_fitted_nuisance.get_pars():", joint_fitted_nuisance.get_pars())
    assert c.tf_all_equal(fitted_log_prob, recalculated_log_prob)
        
def test_hessian_shape(hessian,Hshape):
    H, g = hessian
    print("H:", H)
    print("Hshape:", Hshape)
    Hshape1 = list(Hshape)
    Hshape1[0] = 1 # Case for no nuisance parameters, so no nuisance fits, so only single Hessian returns.
    assert H.shape == Hshape or H.shape == Hshape1

def test_sub_hessian_shapes(decomposed_hessian,sub_hessian_shapes):
    Hii, Hnn, Hin = decomposed_hessian
    Hii_shape, Hnn_shape, Hin_shape = sub_hessian_shapes
    print("Hii:", Hii)
    print("Hnn:", Hnn)
    print("Hin:", Hin)
    if Hii_shape is None: assert Hii == None
    else: assert Hii.shape==Hii_shape
    if Hnn_shape is None: assert Hnn == None
    else: assert Hnn.shape==Hnn_shape
    if Hin_shape is None: assert Hin == None
    else: assert Hin.shape==Hin_shape

def test_quad_prep(quad_prep,joint0,samples):
    A = quad_prep["A"]
    B = quad_prep["B"]
    interest = quad_prep["interest"]
    nuisance = quad_prep["nuisance"]

    # Check if there are any nuisance parameters
    # (note, just counting the tensors, doesn't consider intrinsic dimension of parameters)
    count_nuisances = 0
    for analysis in nuisance.values():
        for parameter in analysis.values():
            count_nuisances += 1

    # Count interest parameters
    count_interest = 0
    for analysis in interest.values():
        for parameter in analysis.values():
            count_interest += 1

    if count_nuisances==0:
        assert A is None
        assert B is None
    else:
        par_shapes = joint0.parameter_shapes()
        s_0, s_batch_shape, s_col_names = c.cat_pars_to_tensor(interest,par_shapes) # stacked interest parameter values at expansion point(s)
        theta_0, t_batch_shape, theta_col_names = c.cat_pars_to_tensor(nuisance,par_shapes) # stacked nuisance parameter values at expansion point(s)
        print("theta_0.shape:", theta_0.shape)
        (n, p) = theta_0.shape
        print("s_batch_shape:", s_batch_shape)
        print("t_batch_shape:", t_batch_shape)
        # Shape requirements:
        # N = number of samples
        # p = number of (scalarized) nuisance parameters
        assert n == N
        assert p == len(theta_col_names)
        assert A.shape == (N,p)
        assert B.shape == (N,p,p)
        ni = len(s_col_names)
        # Can have one expansion point per sample, or just one expansion point overall
        assert s_0.shape == (N,ni) or s_0.shape == (ni,)

        # Recreate the rest of the theta_prof calculation, and test shapes
        # ----------------
        # Need a test "signal", or hypothesis, to do this.
        # Can just use the expasion point for this, but it cancels out
        # the second term of the calculation. So let's just change it a
        # little and hope that the result is still valid for the underlying
        # model.
        s = 1.001*s_0[0:1] # Use 0th sample, keeping the axis 
        print("s_0.shape:", s_0.shape)
        print("s.shape:", s.shape)
        
        # Shift due to A vector:
        Ashift = theta_0 - A
        print("Ashift.shape:", Ashift.shape)

        # Next need to consider shape of s - s_0
        # This is a little complicated, because there are two possibilities:
        # 1. For each hypothesis in s, we may want to evaluate its likelihood under *every* sample
        # 2. We may have one test hypothesis for each sample (e.g. in case we are using "quad" to improve on some fit result)
        #    In this second case, the size of s needs to match the number of samples.
        #
        # But how do detect which situation we are in?
        # If s.shape = (N,ni), should assign one element of s per sample
        # If s.shape = (nH,ni), should evaluate all hypotheses for all samples
        # but this leads to ambiguity if nH==N.
        # So need something more robust.
        # Need to use broadcasting of batch_shapes. There is a batch_shape for the samples/fitted parameters,
        # and another batch_shape for the input hypothesis list. Need to use these to infer broadcasting
        # behaviour.
        # Essentially it is the "all hypotheses for all samples" case which will need the more complicated
        # shape. 
        
        # Should be able to use "normal" rules for log_prob evaluation?

        # 1. sample.shape is compared to (batch_shape,event_shape). Let len(batch_shape,event_shape)=n. 
        #    If sample.shape has fewer dimensions, pad them on the left with singleton dimensions.
        # 2. Broadcast the n right-most dimensions of sample against (batch_shape,event_shape). The remaining
        #    dimensions are interpreted as sample_shape.
        # 3. Result will have shape (sample_shape,batch_shape), since each whole event_shape part is used for
        #    one log_prob evaluation.
        
        # 1. We want to evaluate at nH points, so look at s.shape to obtain batch_shape, which plays the role of
        #    the "distribution" batch_shape.
        #
        # 2. s_0 matches with the sample here, so numpy-style broadcast the batch_shape of s_0 against s.batch_shape
        #    any "leftover" dims on the left of the sample dims become the "sample_shape" dims.
        #
        # 3. Do computation such that resulting log_prob_quad shape is (sample_shape,batch_shape)
        #
        # In this paradigm, to achieve the scenario (1) and (2) results, we need the following input shapes:
        # (1) s.shape               = (N,ni)   "distribution"
        #     s_0.shape             = (N,ni)   "samples"
        #     output log_prob.shape = (N,)     "log_prob"
        #
        # (2) s.shape               =   (nH,ni)  "distribution"
        #     s_0.shape             = (N,1, ni)  "samples"
        #     output log_prob.shape = (N,nH)     "log_prob"
        #
        # So for scenario 2 we need to know how to get those extra "nH" batch dimensions into s_0. This needs to be done
        # when the corresponding JointDistribution is set up, so we have to do it by changing the shape of the parameters
        # input to that distribution.


        # The two situations that we want to work smoothly are:
        # 1. Evaluate likelihoods for a list of hypothesis for all samples

        # 2. Fit nuisance parameters to all samples
        #    log_prob, joint_fitted_nuisance, fitted_pars_nuisance = joint0.fit_nuisance(samples)
        #    # Parameters have shape (N,par_shape)
        #    #print("fitted_pars_nuisance['all']:", fitted_pars_nuisance['all'])
        #    f = joint_fitted_nuisance.log_prob_quad_f(samples)
        #    log_prob_quad = f(fitted_pars_nuisance['fixed']) # the 'fixed' parameters include the 'signal' ones
        #    # Expect to get back log_prob_quad.shape = (N,), i.e. one for each fitted parameter


        # So, to make all this work, shapes should be:
        # N - number/shape of samples
        # nt - number of nuisance parameters (1D, flattened)
        # ni - number of 'interest' parameters (1D, flattened)
        # 
        # A.shape = (N,nt)
        # B.shape = (N,nt,ni)
        # s_0.shape = (N, ni) # Can expand around same point for each sample, or different point (e.g. global best fit)
        #          or (N, 1, ni) # all test points to be used for all expansion points/samples
        # s.shape = (nH,ni)
        #        or (ni,) if only one test point needed 
        #           
        # Broadcasting behaviour is best controlled by making sure s_0 has the desired shape.

        # (s - s_0) = shape determined by bcast rules

        sdiff_shape = c.get_bcast_shape(s.shape,s_0.shape)
        sdiff = tf.broadcast_to(s,sdiff_shape) - tf.broadcast_to(s_0,sdiff_shape)
        print("sdiff.shape:", sdiff.shape)

        # We also need to ensure "B" broadcasts properly against this

        # result of multiplying with matrix B
        print("B.shape:", B.shape)
        Bvec = tf.linalg.matvec(B,sdiff)
        print("Bvec.shape:", Bvec.shape)

        theta_prof = Ashift - Bvec
        
        print("theta_prof.shape:", theta_prof.shape)
        #assert theta_prof.shape == theta_0.shape

        batch_shape = theta_prof.shape[:-1] # Last dimension is flattened parameters, the rest can be thought of as the batch shape

        # de-stack analytically profiled nuisance parameters
        theta_prof_dict = c.decat_tensor_to_pars(theta_prof,nuisance,par_shapes,batch_shape) 
    
def test_fitted_pars(joint_fitted_nuisance,fitted_pars):
    """Check that parameters returned from fit match those in the accompanying "fitted" distribution"""
    print("fitted_pars['all']:", fitted_pars['all'])
    print("joint_fitted_nuisance.get_pars():",joint_fitted_nuisance.get_pars()) 
    assert c.deep_all_equal(fitted_pars['all'], joint_fitted_nuisance.get_pars())

def test_quad_logl_nuisance(joint_fitted_nuisance,fitted_pars,quad_prep):
    """Check that quad-estimate nuisance parameters match fitted ones at the expansion point.
    
       NOTE: As it turns out, values may not exactly match. This is because when fits are
       done numerically they will not find the exact log_prob maxima. Therefore, when expanding
       analytically and profiling around this point, the expansion will quite likely find a 
       slightly *better* log_prob point, with slightly different nuisance parameter values.
       It is kind of hard to know exactly what is tolerable here; the tolerance on the log_prob
       difference (in test_quad_logl) is more informative, in general.
       Here we just have to set a quite loose tolerance to account for this possibility
    """
    signal = fitted_pars["fixed"]
    quad_nuis_pars = joint_fitted_nuisance._nuisance_quad(signal,**quad_prep)
    nuis_pars = fitted_pars["fitted"]
    print("signal:", signal)
    print("quad_nuis_pars:", quad_nuis_pars)
    print("nuis_pars:", nuis_pars)
    if quad_prep["A"] is None:
        assert quad_nuis_pars is None
    else:
        tol = 1e-3
        frac_tol = 0.1 # TODO: kind of loose, why can't we tighten it? Something wrong?
        print("quad_nuis_pars - nuis_pars:", c.deep_minus(quad_nuis_pars,nuis_pars))
        assert c.deep_all_equal_frac_tol(quad_nuis_pars,nuis_pars,frac_tol=frac_tol,fallback_tol=tol)

def test_quad_logl(joint_fitted_nuisance,fitted_log_prob,fitted_pars,quad_prep,samples):
    """Check that shapes for log_prob_quad calculation make sense, and that values match
       the true log_prob at the expansion point
       
       NOTE: As it turns out, values may not exactly match. This is because when fits are
       done numerically they will not find the exact log_prob maxima. Therefore, when expanding
       analytically and profiling around this point, the expansion will quite likely find a 
       slightly *better* log_prob point. We therefore need some extra tolerance to allow for
       this, but only in one direction. Should never get a *worse* result from the expansion,
       unless the likelihood function is absurdly non-quadratic even very close to the maxima."""
    #A, B, interest, nuisance = quad_prep
    #q = joint_fitted_nuisance.neg2loglike_quad(internal_pars,A,B,interest,nuisance,samples)

    # First make sure we understand the fitted parameters in the joint distribution
    # Obtain the quadratic estimate of the log_prob
    f = joint_fitted_nuisance.log_prob_quad_f(samples)
    log_prob_quad = f(fitted_pars['fixed']) # the 'fixed' parameters include the 'signal' ones
    log_prob = fitted_log_prob

    # Test shapes. 
    print("samples:", samples)
    print("fitted_pars['fixed']:", fitted_pars['fixed'])
    print("log_prob_quad.shape:",             log_prob_quad.shape)
    print("c.deep_size(samples):",            c.deep_size(samples))
    print("c.deep_size(fitted_pars['fixed']):", c.deep_size(fitted_pars['fixed']))

    assert log_prob_quad.shape[0]==c.deep_size(samples)

    # Test values; quad log_prob should equal true log_prob at the expansion points. And we
    # have only calculated it at the expansion points, so all values should be equal
    print("log_prob_quad:", log_prob_quad)
    print("fitted_log_prob:", fitted_log_prob)
    print("log_prob_quad - fitted_log_prob:", log_prob_quad - fitted_log_prob)
    #assert c.tf_all_equal(log_prob_quad, fitted_log_prob)
    tol_tight = 1e-6
    tol_loose = 1e-3
    assert tf.reduce_all(tf.less_equal(log_prob_quad - fitted_log_prob,tol_loose)) # Looser tolerance for log_prob_quad > fitted_log_prob
    assert tf.reduce_all(tf.less_equal(-(log_prob_quad - fitted_log_prob),tol_tight)) # Tighter tolerance for log_prob_quad < fitted_log_prob; should only be possible via floating point error

# @pytest.mark.parametrize(
#     "analysis,single_hypothesis,hypothesis_list,Hessian_list",
#     [(get_obj(name), get_test_hypothesis(name), get_hypothesis_lists(name), H_shape) for name,H_shape in Hessians_shape_list.items()],
#     ids = Hessians_shape_list.keys()
#     )
# def test_Hessian_shape_single_with_list(analysis,single_hypothesis,hypothesis_list,Hessian_list):
#     """Compute Hessian for list of parameters, for samples generated
#        from a single hypothesis"""
#     in_pars = {analysis.name: single_hypothesis}
#     joint = JointDistribution([analysis],in_pars)
#     x = joint.sample(4)
#     print("x:", x)
#     test_pars = {analysis.name: hypothesis_list}
#     print("test_pars:", test_pars)
#     H = joint.Hessian(test_pars,x)
#     print("H:", H)
#     #assert False

