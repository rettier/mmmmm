from keras.engine.saving import load_model
import numpy as np
from train import MaskedDense, mask0

#model = load_model("models/7x5x11-pruned.h5", custom_objects={"MaskedDense": MaskedDense})
model = load_model("models/7x5x11-pruned.h5", custom_objects={"MaskedDense": MaskedDense})
w = model.get_weights()[0]
#w = np.multiply(w, mask0)
print(w)
w[np.abs(w) <= 0.1] = 0
w[np.abs(w) > 0.1] = 1
print(w.astype(np.uint8))

