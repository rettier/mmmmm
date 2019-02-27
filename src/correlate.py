import numpy as np
from train import load_inputs_outputs, idx

if __name__ == "__main__":
    inputs, outputs, iterations = load_inputs_outputs(threads=1024)

    cols = (
        ("max", idx.max),
        ("min", idx.min),
        ("sum", idx.sum),
        ("rms", idx.rms),
        ("cols", idx.cols),
        ("avg", idx.avg),
    )


    results = []
    for name1, idx1 in cols:
        for name2, idx2 in cols:
            for op in [np.add, np.subtract, np.multiply, np.divide]:
                a = op(inputs[:, idx1], inputs[:, idx2])
                mask = np.logical_not(np.isnan(a))

                a = a[mask]
                b = outputs[mask]
                n = outputs.shape[0]
                sb = np.var(b)
                mb = np.mean(b)
                sa = np.var(a)
                ma = np.mean(a)

                import math
                if math.isnan(sa) or math.isnan(sb) or sa == 0 or sb == 0:
                    continue

                cor = 1. / (n - 1) * np.sum(np.multiply((b - mb), (a - ma))) / (sa * sb)
                results.append((name1, str(op).replace("<ufunc '", ""), name2, cor))

    results = sorted(results, key=lambda x: abs(float(x[3])))
    for x in results:
        print(x)
