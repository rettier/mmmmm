from datetime import datetime

import keras
import numpy as np
from keras import Sequential
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.engine.saving import load_model
from keras.layers import Dense, Lambda
from sklearn.model_selection import train_test_split
import tensorflow as tf

model = load_model("src/models/7x5x11-full.h5")
arr = np.array([[1902.02], [475.0], [2378.0], [476.0], [264.0], [52.0], [9.0]])
# arr = np.array([[787.0], [358.0], [1146.0], [359.0], [127.0], [39.0], [9.0]])
print(model.predict(arr.T))
