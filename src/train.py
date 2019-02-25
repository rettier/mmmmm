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


def load_inputs_outputs():
    return np.load("inputs.npy"), np.load("outputs.npy")

mask0 = [[1, 1, 1, 0, 1],
         [0, 0, 0, 1, 1],
         [0, 1, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 1, 1, 1],
         [0, 1, 1, 0, 0],
         [0, 1, 1, 0, 1]]

# mask0 = np.ones((7,5))


class MaskedDense(Dense):
    def __init__(self, *args, **kwargs):
        self.weight_mask = tf.constant(mask0, dtype=np.float32)
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        output = K.dot(inputs, tf.multiply(self.kernel, self.weight_mask))
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output



def create_masked():
    model = Sequential()
    model.add(MaskedDense(5, weight_mask=mask0, kernel_initializer='normal', activation='relu'))
    model.add(Dense(11, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_smaller():
    model = Sequential()
    model.add(Dense(5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(11, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    inputs, outputs = load_inputs_outputs()

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.10, random_state=0)
    y_train = keras.utils.to_categorical(y_train, num_classes=11)
    y_test = keras.utils.to_categorical(y_test, num_classes=11)

    values = X_train[:, -1]
    mean_val = np.average(values)

    batch_size = 128
    tb = TensorBoard(
        log_dir="logs/{}".format(datetime.now().strftime("%H:%M:%S")),
        update_freq=batch_size * 100
    )

    model = create_masked()

    #o_model = load_model("test")
    #model.set_weights(o_model.get_weights())

    model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_test, y_test),
        callbacks=[tb],
        epochs=10,
        batch_size=batch_size,
    )

    model.save("test2")
