import math
import os

from train import MaskedDense

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random
from functools import partial
from multiprocessing.pool import Pool

import numpy as np
from tqdm import tqdm

from loadsim import calculate_iterations
from stats import list_mats


def _load(f):
    stats = np.load(f + ".stats.npy")[()]
    best = np.load(f + ".best.npy")[()]
    calculations = stats["calculations"]
    del stats["calculations"]
    result = {"stats": stats["stats"], "calculations": calculations, "best": best, "name": os.path.basename(f)}
    return result


def load_all(count=None, fraction=1.0):
    files = list_mats(fraction, suffix=".best.npy")
    if count is not None:
        files = files[:count]
    pool = Pool(processes=16)
    try:
        return list(tqdm(pool.imap_unordered(_load, files), total=len(files)))
    finally:
        pool.close()


def decide_16(stats, row):
    return 16


def decide_32(stats, row):
    return 32


def decide_1024(stats, row):
    return 1024


def decide_matze(stats, row):
    stats = stats["stats"][row]
    maxOperationsPerCol = stats["max"]
    lastMinusStart = stats["rows"]
    sumOperations = stats["sum"]
    THREADS = 1024

    opsPerNnz = max(1, sumOperations / lastMinusStart)
    threads = int(max(0, min(10, math.log2(opsPerNnz) - 1)))
    if maxOperationsPerCol > (1 << threads) * 80 and lastMinusStart < 64:
        threads += 3
    elif maxOperationsPerCol > (1 << threads) * 40 and lastMinusStart < 128:
        threads += 2
    elif maxOperationsPerCol > (1 << threads) * 20 and lastMinusStart < 256:
        threads += 1

    if lastMinusStart * 4 > 10 and threads >= 6:
        threads -= 1
    if lastMinusStart * 2 > 10 and threads >= 5:
        threads -= 1

    totalConcurrentops = lastMinusStart * (1 << threads)
    if totalConcurrentops < THREADS >> 2:
        threads += 1
    elif totalConcurrentops < THREADS >> 1:
        threads += 1

    return min(THREADS, (1 << threads))


input_keys = ['max', 'min', 'sum', 'sum_without_max', 'avg', 'avg_without_max', 'rows']
threadcount = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
thread_loookup = {x: threadcount[x] for x in range(11)}
model = None


def decide_nn_full(stats, row):
    global model
    if model is None:
        from keras.engine.saving import load_model
        model = load_model("models/7x5x11-full.h5")
    stats = stats["stats"][row]
    input = np.asarray([[stats[x] for x in input_keys], ])
    return thread_loookup[np.argmax(model.predict([input], batch_size=1))]

def decide_nn_pruned(stats, row):
    global model
    if model is None:
        from keras.engine.saving import load_model
        model = load_model("models/7x5x11-pruned.h5", custom_objects={"MaskedDense": MaskedDense})
    stats = stats["stats"][row]
    input = np.asarray([[stats[x] for x in input_keys], ])
    return thread_loookup[np.argmax(model.predict([input], batch_size=1))]


def eval(func, mat):
    best_count = 0
    my_count = 0
    for row, calculations in mat["calculations"].items():
        if mat["best"][row]["iterations"] is None:  # bug earlier when nzz > 1024
            continue
        best_count += mat["best"][row]["iterations"]
        threads = func(mat, row)
        my_count += calculate_iterations(calculations, threads, threads_available=1024)
    return mat["name"], best_count, my_count


pool = None


def multiproc_eval(func, mats):
    global pool
    if pool is None:
        pool = Pool(processes=20)
    return list(tqdm(pool.imap_unordered(partial(eval, func), mats), total=len(mats)))


def singleproc_eval(func, mats):
    return [eval(func, mat) for mat in mats]


def write_results(results, name):
    results = sorted(results, key=lambda x: x[0])
    with open(name + ".csv", "w") as f:
        f.writelines("\n".join(str(x) for x in results))

    my_iters = sum(x[2] for x in results)
    ideal_iters = sum(x[1] for x in results)
    print("% iterations", 100 * float(my_iters) / ideal_iters)


if __name__ == "__main__":
    random.seed(1)
    mats = load_all(fraction=1)

    write_results(multiproc_eval(decide_nn_pruned, mats), "nnpruned")
    write_results(multiproc_eval(decide_nn_full, mats), "nnfull")
    write_results(multiproc_eval(decide_matze, mats), "matze")
    write_results(multiproc_eval(decide_32, mats), "32")

    """
    % iterations  113.38444601101061  8:41 nn
    % iterations  218.35587866858393  9:12 matze
    % iterations  457.11604754581776  4:10 32
    """

