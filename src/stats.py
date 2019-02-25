import glob
import os
import random
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import tqdm
from scipy.io import mmread


def load_mat(path):
    prep_file = path + ".prep.npy"
    if os.path.exists(prep_file):
        return np.load(path + ".prep.npy")[()]
    else:
        return read_stats(path)


def matze_row_stats(mat):
    row_calculations = defaultdict(list)
    for row, col in zip(*mat["nonzero"]):
        row_calculations[row].append(mat["nz_per_row"][col])

    for row, calculations in row_calculations.items():
        row_calculations[row] = np.asarray(calculations, dtype=np.int32)

    row_stats = {}
    for row, values in row_calculations.items():
        sum = np.sum(values)
        max = np.max(values)
        min = np.min(values)
        mean = np.average(values)

        if min == max:  # all values are the same
            mean_nonmax = mean
        else:
            mean_nonmax = np.average(values[values != max])

        row_stats[row] = {
            "max": max,
            "min": min,
            "sum": sum,
            "sum_without_max": sum - max,
            "avg": mean,
            "avg_without_max": mean_nonmax,
            "rows": len(values),
        }

    return {
        "calculations": row_calculations,
        "stats": row_stats
    }



def read_stats(path):
    try:
        with open(path, "rb") as f:
            mat = mmread(f)
    except:
        os.unlink(path)
        print("error reading {}".format(path))
        return

    rows, cols = mat.shape

    nz_per_row = np.asarray(mat.getnnz(axis=0), dtype=np.int32)
    assert len(nz_per_row) == cols  # for matrix b, since its transposed

    result = {
        "name": os.path.basename(path),
        "nz_per_row": nz_per_row,
        "nonzero": mat.nonzero()
    }

    np.save(path + ".prep.npy", np.asarray(result))
    return result


def load_and_stat(p):
    stats_path = p + ".stats.npy"
    if os.path.exists(stats_path):
        return np.load(stats_path)[()]
    else:
        mat = load_mat(p)
        with_stats = matze_row_stats(mat)
        np.save(stats_path, np.asarray(with_stats))
        return with_stats


def list_mats(fraction=1.0, suffix=""):
    files = glob.glob("../mat/*.mtx{}".format(suffix))
    if fraction != 1.0:
        count = max(int(len(files) * fraction), 1)
        files = random.choices(files, k=count)

    files = [(x, os.path.getsize(x)) for x in files]
    files.sort(key=lambda filename: filename[1])
    return [x[0].replace(suffix, "") for x in files]


def load_all_with_stats(fraction=1.0):
    files = list_mats(fraction)
    pool = Pool(processes=16)
    try:
        return tqdm.tqdm(pool.imap_unordered(load_and_stat, files), total=len(files))
    finally:
        pool.close()


if __name__ == "__main__":
    load_all_with_stats(fraction=0.1)
