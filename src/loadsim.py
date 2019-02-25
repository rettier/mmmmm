import os
import random
from multiprocessing.pool import Pool
from pprint import pprint

import numpy as np
from tqdm import tqdm

from src.stats import load_all_with_stats, list_mats, load_and_stat


def calculate_iterations(calculations, threads, threads_available):
    blocks = threads_available // threads
    calc_blocks = np.maximum(np.ceil(calculations / threads), 1)
    sums = [np.sum(calc_blocks[block::blocks]) for block in range(blocks)]
    return np.max(sums)


def calculate_best(mat, threads_available=1024, withstats=True):
    assert threads_available <= 4096, "extend array below because this is bad code"
    threads = [x for x in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] if x <= threads_available]
    result = dict()

    if withstats:
        stats = mat["stats"]

    for row, calculations in mat["calculations"].items():
        if withstats:
            min, max = stats[row]["min"], stats[row]["max"]
        else:
            min, max = 0, 1024

        best_iterations = None
        best_thread_count = None

        for thread_count in threads:
            if thread_count < min and thread_count != threads[-1]:
                continue

            iterations = calculate_iterations(calculations, thread_count, threads_available)
            if best_iterations is None or iterations < best_iterations:
                best_iterations = iterations
                best_thread_count = thread_count

        result[row] = {
            "threads": best_thread_count,
            "iterations": best_iterations
        }

    return result


def process_best(f):
    path = f + ".best.npy"
    if os.path.exists(path):
        return

    mat = load_and_stat(f)
    result = calculate_best(mat)
    np.save(path, np.asarray(result))


if __name__ == "__main__":
    random.seed(0)

    files = list_mats(fraction=1.0)
    pool = Pool(processes=16)
    try:
        for _ in tqdm(pool.imap_unordered(process_best, files), total=len(files)):
            pass
    finally:
        pool.close()

    # matze example 1
    # calculations = {
    #     0: np.asarray([2000, ] + [10] * 28),
    #     1: np.asarray([20000, 20000, 5000, 1000, 100] + [10] * 22),
    #     2: np.asarray([1024, 1024, 1024, 1024])
    # }
    # pprint(calculate_best({"calculations": calculations}, withstats=False))
