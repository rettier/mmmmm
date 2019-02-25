from multiprocessing.pool import Pool

import numpy as np
from tqdm import tqdm

from stats import list_mats


def load_best_and_stats(f):
    return np.load(f + ".train.npy")[()]


def load_all(count=None, fraction=1.0):
    files = list_mats(fraction=fraction, suffix=".best.npy")
    if count is not None:
        files = files[:count]
    pool = Pool(processes=16)
    try:
        return list(tqdm(pool.imap_unordered(load_best_and_stats, files), total=len(files)))
    finally:
        pool.close()


def _generate_reduced_data(f):
    stats = np.load(f + ".stats.npy")[()]
    best = np.load(f + ".best.npy")[()]

    calculations = stats["calculations"]
    for row, calcs in calculations.items():
        calculations[row] = len(calcs)

    del stats["calculations"]
    result = {"stats": stats["stats"], "calculations": calculations, "best": best}
    np.save(f + ".train.npy", result)


def generate_reduced_data(count=None):
    files = list_mats(fraction=1.0, suffix=".best.npy")
    if count is not None:
        files = files[:count]
    pool = Pool(processes=16)
    try:
        return list(tqdm(pool.imap_unordered(_generate_reduced_data, files), total=len(files)))
    finally:
        pool.close()


def generate_input_and_output(data):
    unrolled = []
    for mat in data:
        stats = mat["stats"]
        best = mat["best"]
        for row in stats:
            row_stats = stats[row]
            row_threadcount = best[row]["threads"]
            if not row_threadcount:  # due to a bug earlier
                continue
            unrolled.append({"in": row_stats, "out": row_threadcount})

    threadcount = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    thread_loookup = {threadcount[x]: x for x in range(11)}

    input_keys = unrolled[0]["in"].keys()
    inputs = np.ndarray(shape=(len(unrolled), len(input_keys)), dtype=np.float32)
    outputs = np.ndarray(shape=(len(unrolled), 1), dtype=np.int32)

    for i, sample in enumerate(unrolled):
        inputs[i, :] = [sample["in"][x] for x in input_keys]
        outputs[i, :] = thread_loookup[sample["out"]]

    return inputs, outputs


if __name__ == "__main__":
    data = load_all()
    inputs, outputs = generate_input_and_output(data)
    np.save("inputs.npy", inputs)
    np.save("outputs.npy", outputs)
