from keras.engine.saving import load_model
import numpy as np
from train import MaskedDense, mask0

model = load_model("test2", custom_objects={"MaskedDense": MaskedDense})
w = model.get_weights()[0]
w = np.multiply(w, mask0)
print(w)
