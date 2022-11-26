import test_all_methods
from utils import INTENSIFICATION_MODELS
from tests import LOCAL_SEARCH


if __name__ == '__main__':
    test_all_methods.main(INTENSIFICATION_MODELS, LOCAL_SEARCH, processes=12, timeout=360)
