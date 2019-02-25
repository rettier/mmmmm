"""
Show the histogram of thread counts and iteration counts
"""

import numpy as np
import os

import matplotlib.pyplot as plt

from stats import list_mats


def load(f):
    path = f + ".best.npy"
    if not os.path.exists(path):
        print("best thread count not calculated for {}".format(f))
        return

    return np.load(path)[()]


if __name__ == "__main__":
    for mat in list_mats():
        best = load(mat)
        if not best:
            continue

        thread_counts = [x["threads"] for x in best.values()]
        iterations = [x["iterations"] for x in best.values()]

        f, (ax1, ax2) = plt.subplots(1, 2)
        f.suptitle(os.path.basename(mat))

        ax1.set_title("thread counts")
        ax1.hist(thread_counts)

        ax2.set_title("iterations counts")
        ax2.hist(iterations)
        f.show()
