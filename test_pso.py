import test_all_methods
from utils import PSO_MODELS
from tests import PSO_SEARCH


if __name__ == '__main__':
    test_all_methods.main(PSO_MODELS, PSO_SEARCH, processes=4, timeout=1000)
