"""Integration tests that apply to all Analysis classes,
   and check that they conform correctly to the
   required interface"""

import pytest

from tensorflow_probability import distributions as tfd
from jmctf import JointDistribution
from jmctf_tests.analysis_class_register import get_id_list, get_obj, get_test_hypothesis, get_hypothesis_lists

# Fixture to create just analysis objects
@pytest.fixture(scope="module", 
                params=get_obj().values(),
                ids = get_id_list())
def analysis(request):
    yield request.param

# Fixture to create analysis objects and provide test parameters to use with them 
@pytest.fixture(scope="module", 
                params=list(zip(get_obj().values(),get_test_hypothesis().values()))
                      +list(zip(get_obj().values(),get_hypothesis_lists().values())),
                ids = [name + " (single hypothesis)" for name in get_id_list()]
                     +[name + " (hypothesis list)" for name in get_id_list()])
def analysis_and_parameters(request):
    analysis, parameters = request.param
    yield (analysis, parameters)

# Fixture to create tensorflow_probability models from each Analysis/test parameter combination
@pytest.fixture(scope="module")
def model(analysis_and_parameters):
    analysis, pars = analysis_and_parameters
    yield analysis.tensorflow_model(pars)

# Fixture to create JointDistribution objects from each Analysis/test parameter combination
@pytest.fixture(scope="module")
def joint(analysis_and_parameters):
    analysis, pars = analysis_and_parameters
    joint = JointDistribution([analysis],{analysis.name: pars})
    yield joint

def test_tensorflow_models(model):
    """tensorflow_model function should return dictionary of tensorflow_probability distribution objects"""
    assert isinstance(model,dict)
    for key,value in model.items():
        assert isinstance(key, str)
        assert isinstance(value, tfd.Distribution)

# Decorator to help run test for all analyses
def do_for_all_analyses(test_func):
    @pytest.mark.parametrize(
        "analysis,single_hypothesis,many_hypotheses",
        zip(get_obj().values(),get_test_hypothesis().values(),get_hypothesis_lists().values()),
        ids = get_id_list()
        )
    def wrapper(analysis,single_hypothesis,many_hypotheses):
        test_func(analysis,single_hypothesis,many_hypotheses)
    return wrapper

def test_JointDistribution_init_basic(analysis):
    """Test that JointDistribution objects can be instantiated from the test analysis objects
       No parmeter version"""
    joint = JointDistribution([analysis])

def test_JointDistribution_init_pars(joint):
    """Test that JointDistribution objects can be instantiated from the test analysis objects
       Fixed parameters version"""
    pass # Just testing that "joint" is created so far.

def test_JointDistribution_output_shapes_single(analysis_and_parameters):
    """Test that flow of shapes through JointDistribution samples -> pdf is correct"""
    analysis, pars = analysis_and_parameters
    joint = JointDistribution([analysis],{analysis.name: pars})
    n = 10
    x = joint.sample(n)
    p = joint.log_prob(x)
    # Input par tensors have dimension (m,...) where m is the number of separate hypotheses
    # (other dimensions relate to 'intrinsic' dimension of each parameter)
    # Sample tensors have dimension (n,m,...) where m " "      " "
    #                                         and n is the number of independent draws from the distribution
    # (other dimensions relate to the 'intrinsic' dimension of a single sample from each component of the distribution)
    # In tensorflow_probability language TODO: figure it out
    # Probability tensor should have dimension (n,m)
    # Here we test the consistency of all these, assuming the functions above have run successfully

    # CORNER CASE
    # If input parameters have dimension (), i.e. are scalars, then the other outputs have a slightly different structure since
    # the extra dimension (that would be a singleton) is dropped:
    # Sample tensors become: (n,...)
    # Probability tensors become: (n)

    # Will only see these print statements when test fails
    print(analysis.name, "pars:", pars)
    print(analysis.name, "x:", x)
    print(analysis.name, "p:", p)

    m = None
    for name, par in pars.items():
        if hasattr(par, "shape"):
            # par is some sort of array-like thing
            if par.shape == ():
                this_m = 0
            else:
                this_m = par.shape[0]
        elif hasattr(par, "__len__") and hasattr(par, '__getitem__'):
            # par is some sort of list type of thing
            this_m = len(par)
        else:
            this_m = 0 # Take it to be a scalar
        if m is None:
            m = this_m
        else:
            assert this_m == m

    for name, xi in x.items():
        assert xi.shape[0] == n
        if m!=0: assert xi.shape[1] == m

    assert p.shape[0] == n
    if m!=0: assert p.shape[1] == m
