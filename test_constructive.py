import test_all_methods
from utils import CONSTRUCTIVE_MODELS
from tests import CONSTRUCTIVE_SEARCH


if __name__ == '__main__':
    test_all_methods.main(CONSTRUCTIVE_MODELS, CONSTRUCTIVE_SEARCH, processes=4)
