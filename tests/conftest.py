import pytest

def pytest_addoption(parser):
    parser.addoption(
        '--dim-state-space',
        type=int,
        dest='S',
        default=100,
        help='Set dimension of the state space. Default: 100',
    )
    parser.addoption(
        '--period-length',
        type=int,
        dest='M',
        default=2,
        help='Set length of period. Default: 2',
    )
    parser.addoption(
        '--interval-length',
        type=int,
        dest='N',
        default=3,
        help='Set length of finite-time interval. Default: 3',
    )
    
@pytest.fixture
def S(request):
    return request.config.getoption('S')

@pytest.fixture
def M(request):
    return request.config.getoption('M')

@pytest.fixture
def N(request):
    return request.config.getoption('N')