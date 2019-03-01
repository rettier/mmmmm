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

model = load_model("src/models/6x5x11-full.h5")

layers = 0
for layer in model.layers:
    layers += 1
    weights = layer.get_weights()
    text = "const int layer" + str(layers) + "inputs = " + str(len(weights[0])) + ";\n"
    text += "const int layer" + str(layers) +"outputs = " + str(len(weights[0][0])) + ";\n"
        
    text += "const float layer" + str(layers) + "weights[" + str(len(weights[0]) * len(weights[0][0])) + "] = {"
    for w in weights[0]:
        for x in w:
            text += str(x) + ", "
        text += "\n\n"
    text += "};\nconst float layer" + str(layers) + "offsets[" + str(len(weights[1])) + "] = {"
    for w in weights[1]:
        text += str(w) + ", "
    text += "};\n"
    print(text)
