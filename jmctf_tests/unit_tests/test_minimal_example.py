import pytest
from jmctf_tests.common_fixtures import square

@pytest.fixture(scope="module",params=[1,2,3,4,5])
def x(request):
    x_val = request.param
    return x_val

def test_square(x,square):
    print("square:", square)
    assert square==x**2
