import os
import time
import inspect
import pandas as pd
import traceback
from itertools import product, repeat
from multiprocessing import Pool, Manager, TimeoutError

from tests import ALL_TESTS
from packer import IrregularPackerPSO


# PROCESSES = 1
PROCESSES = 12
TIMEOUT = 70
OUTPUT_FILE = 'results/test_results.csv'


def test_model(model, test, kwargs, results):
    try:
        t0 = time.perf_counter()
        items_str = 'radius' if hasattr(model, 'radius') else 'shapes'

        base_class = inspect.getclasstree([model])
        args = inspect.getfullargspec(base_class[0][0].__init__).args
        args = list(set(
            inspect.getfullargspec(model.__init__).args +
            inspect.getfullargspec(model.__init__).kwonlyargs +
            args)
        )
        kwargs = {arg: kwargs[arg] for arg in kwargs if arg in args}
        packer = model(test['container'], test[items_str], **kwargs)
        packer.pack()

        if issubclass(type(packer), IrregularPackerPSO):
            score = packer.global_optimum
        else:
            score = packer.score()

        t1 = time.perf_counter()

        result = {
            'test': test['name'],
            'model': model.__name__,
            **kwargs,
            'time': t1 - t0,
            'score': score
        }
        print(result)
        results.append(result)

    except Exception as ex:
        print(ex)
        traceback.print_exc()


def main(models, search, processes=PROCESSES, timeout=TIMEOUT):
    combinations = list(product(*search.values()))
    # processed_tests = []
    for combi in combinations:
        for test in ALL_TESTS:
            for i in range(0, len(models), PROCESSES):
                kwargs = dict(zip(search.keys(), combi))

                # current_test = [
                #     model.__name__,
                #     *kwargs.values()
                # ]
                # if current_test in processed_tests:
                #     continue

                mgr = Manager()
                results = mgr.list()
                pool = Pool(processes=processes)
                result = pool.starmap_async(
                    test_model,
                    zip(
                        # ALL_TESTS[:1],
                        models[i:i + processes],
                        repeat(test),
                        repeat(kwargs),
                        repeat(results)
                    )
                )

                try:
                    result.get(timeout=timeout)
                    pool.close()
                    pool.join()
                except TimeoutError:
                    pool.close()
                    pool.terminate()

                # processed_tests.append(current_test)

                df = pd.DataFrame(list(results))
                if not os.path.isfile(OUTPUT_FILE):
                    df.to_csv(OUTPUT_FILE, index=False)
                else:
                    df_old = pd.read_csv(OUTPUT_FILE)
                    df = pd.concat([df_old, df])
                    df.to_csv(OUTPUT_FILE, index=False)
