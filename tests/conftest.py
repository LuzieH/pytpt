import pytest

def pytest_addoption(parser):
    parser.addoption(
        '--dim-state-space',
        type=int,
        dest='S',
        default=100,
        help='Set dimension of the state space. Default: 100',
    )

@pytest.fixture
def S(request):
    return request.config.getoption('S')
