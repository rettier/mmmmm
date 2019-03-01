import math
import os

from train import MaskedDense, load_inputs_outputs, idx, custom_loss, topk

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


def decide(count, *args):
    return count


def decide_matze(stats):
    maxOperationsPerCol = stats[idx.max]
    lastMinusStart = stats[idx.cols]
    sumOperations = stats[idx.sum]

    opsPerNnz = max(1, sumOperations / lastMinusStart)
    threads = int(max(0, min(THREADS_LOG2, math.log2(opsPerNnz) - 1)))
    if maxOperationsPerCol > (1 << threads) * 80 and lastMinusStart < 64:
        threads += 3
    elif maxOperationsPerCol > (1 << threads) * 40 and lastMinusStart < 128:
        threads += 2
    elif maxOperationsPerCol > (1 << threads) * 20 and lastMinusStart < 256:
        threads += 1

    if lastMinusStart * 16 > THREADS_LOG2 and threads >= 7:
        threads -= 1
    if lastMinusStart * 8 > THREADS_LOG2 and threads >= 5:
        threads -= 1

    totalConcurrentops = lastMinusStart * (1 << threads)
    if totalConcurrentops < THREADS >> 2:
        threads += 2
    elif totalConcurrentops < THREADS >> 1:
        threads += 1

    return min(THREADS, (1 << threads))


def decide_super_matze(stats):
    maxOperationsPerCol = stats[idx.max]
    cols = stats[idx.cols]
    sumOps = stats[idx.sum]
    sumOps2 = stats[idx.sum] - stats[idx.max]

    opsPerNnz = max(1.0, sumOps2 / cols)
    threads = round(max(0, min(THREADS_LOG2, math.log2(opsPerNnz))))
    threads += min(THREADS_LOG2 - threads, int(maxOperationsPerCol * 0.8 / (1 << threads)))
    if threads >= 4:
        threads -= min(2, cols // threads // threads)

    threads = max(0, threads)
    totalConcurrentops = cols * (1 << threads)
    threads += min(3, (THREADS >> 2) // totalConcurrentops)

    return min(THREADS, (1 << threads))
    return min(THREADS, (1 << threads))


input_keys = ['max', 'min', 'sum', 'sum_without_max', 'avg', 'avg_without_max', 'rows']
input_keys = ['max', 'sum', 'sum_without_max', 'avg', 'avg_without_max', 'rows']
threadcount = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
thread_loookup = {x: threadcount[x] for x in range(11)}
model = None


def decide_nn(model_path, stats):
    global model
    if model is None:
        from keras.engine.saving import load_model
        model = load_model(model_path, custom_objects={
            "custom_loss": custom_loss,
            "top3_acc": topk
        })
    return thread_loookup[np.argmax(model.predict(np.asarray([stats[:18]]), batch_size=1))]


def decide_nn_pruned(stats, row):
    global model
    if model is None:
        from keras.engine.saving import load_model
        model = load_model("models/7x5x11-pruned.h5", custom_objects={"MaskedDense": MaskedDense})
    stats = stats["stats"][row]
    input = np.asarray([[stats[x] for x in input_keys], ])
    return thread_loookup[np.argmax(model.predict([input], batch_size=1))]


def eval_old(func, mat):
    best_count = 0
    my_count = 0
    for row, calculations in mat["calculations"].items():
        if mat["best"][row]["iterations"] is None:  # bug earlier when nzz > 1024
            continue
        best_count += mat["best"][row]["iterations"]
        threads = func(mat, row)
        my_count += calculate_iterations(calculations, threads, threads_available=1024)
    return mat["name"], best_count, my_count


def eval(func, i):
    thread_idx = int(math.log2(func(inputs[i, ...])))
    return thread_idx, np.min(iterations[i, ...]), iterations[i, thread_idx]


pool = None


def multiproc_eval(func, ):
    count = inputs.shape[0]
    global pool
    if pool is None:
        pool = Pool(processes=20)
    return list(tqdm(pool.imap_unordered(partial(eval, func), range(count)), total=count))


def singleproc_eval(func,):
    return [eval(func, i) for i in range(len(inputs))]


def write_results(results, name):
    results = sorted(results, key=lambda x: x[0])
    with open(name + ".csv", "w") as f:
        f.writelines("\n".join(str(x) for x in results))

    my_iters = sum(x[2] for x in results)
    ideal_iters = sum(x[1] for x in results)
    arr = np.asarray(results)
    my_iters_squared = sum(x[2]*x[2] for x in results)
    ideal_iters_squared = sum(x[1]*x[1] for x in results)
    

    # import matplotlib.pyplot as plt
    # plt.title("func")
    # plt.hist(list(x[0] for x in results))
    # plt.show()

    # plt.title("ideal")
    # plt.hist(outputs)
    # plt.show()

    print("{} iterations, {} square iterations".format(100 * float(my_iters) /
                                                       ideal_iters, 100 * math.sqrt(float(my_iters_squared) / ideal_iters_squared)))


def decide_super_matze_2(stats):
    # return 32
    # stats[idx.max] = 20
    # stats[idx.sum] = 206
    # stats[idx.cols] = 14
    # stats[idx.max] = 50
    # stats[idx.sum] = 1950
    # stats[idx.cols] = 45
    maxOpsPerCol = int(stats[idx.max])
    cols = int(stats[idx.cols])
    sumOps = int(stats[idx.sum])
    sumOps2 = int(stats[idx.sum] - stats[idx.max])

    opsPerNnz = max(1.0, sumOps2 / cols)
    avg_threads = round(max(0, min(THREADS_LOG2, math.log2(opsPerNnz))))
    threads = avg_threads
    colIters = math.ceil(cols / (THREADS / (1 << threads)))
    maxIters = math.ceil(maxOpsPerCol / (1 << threads))
    if maxIters > colIters * 2:
        threads += min(THREADS_LOG2 - threads, max(1, int(math.log2(maxIters // colIters // 2))))

    colIters = math.ceil(cols / (THREADS / (1 << threads)))
    maxIters = math.ceil(maxOpsPerCol / (1 << threads))
    if (1 << threads) * 2 > (sumOps2 / cols) and colIters > maxIters * 2:
        threads -= min(threads // 2, max(1, int(math.log2(colIters // maxIters // 2))))

    threads = max(0, threads)
    concurrentOps = cols << threads

    if concurrentOps < THREADS:
        threads += int(math.log2(THREADS / concurrentOps))

    threads = min(THREADS_LOG2, threads)
    threads = max(0, threads)

    optimum = min(stats[5:])
    myIter = stats[5+threads]

    if myIter >= optimum * 2:
        print("{}% more iterations than best case".format(100*myIter/optimum))

    return min(THREADS, (1 << threads))

THREADS = 256
THREADS_LOG2 = int(np.log2(THREADS))
if __name__ == "__main__":
    random.seed(1)
    # mats = load_all(fraction=1)
    count = 10000
    inputs, outputs, iterations = load_inputs_outputs(threads=THREADS, max_count=count, load_1M=True)

    if True:
        inputs = np.load(
            "/media/mathias/Data/trainingdata/{}_1M.npy".format(THREADS))
        inputs = inputs[inputs[:, idx.cols] != 0]
        inputs = inputs[inputs[:, idx.avg] != 0]
        inputs = inputs[inputs[:, idx.max] != 0]
        steps = max(1, int(np.floor(len(inputs) / count)))
        inputs = inputs[::steps, ...]
        write_results(singleproc_eval(decide_super_matze_2), "super-matze")
    else:
        write_results(singleproc_eval(
            partial(decide_nn, "{}.h5".format(THREADS))), "nn")


    # global model
    # if model is None:
    #     from keras.engine.saving import load_model
    #     model = load_model(model_path, custom_objects={
    #         "custom_loss": custom_loss,
    #         "top3_acc": topk
    #     })
    # return thread_loookup[np.argmax(model.predict(np.asarray([stats[:18]]), batch_size=1))]

    # write_results(multiproc_eval(decide_nn_pruned, mats), "nnpruned")
    # write_results(multiproc_eval(decide_matze), "matze")
    # write_results(multiproc_eval(partial(decide, 32)), "32")

    """
    % iterations  113.38444601101061  8:41 nn
    % iterations  218.35587866858393  9:12 matze
    % iterations  457.11604754581776  4:10 32
    """
