import functools
from datetime import datetime

import keras
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, Softmax
from keras.metrics import top_k_categorical_accuracy
from sklearn.model_selection import train_test_split


class idx:
    max = 0
    min = 1
    sum = 2
    rms = 3
    cols = 4
    avg = 5
    avg_without_max = 6
    sum_without_max = 7

    in_out_split = 5

    t1 = 0
    t2 = 1
    t4 = 2
    t8 = 3
    t16 = 4
    t32 = 5
    t64 = 6
    t128 = 7
    t256 = 8
    t512 = 9
    t1024 = 10


def load_inputs_outputs(threads=64, max_count=4000000):
    data = np.load("/media/mathias/Data/trainingdata/{}.npy".format(threads))
    data = data[data[:, idx.cols] != 0]
    data = data[data[:, idx.avg] != 0]
    data = data[data[:, idx.max] != 0]
    steps = max(1, int(np.floor(len(data) / max_count)))
    data = data[::steps, ...]

    inputs = np.ndarray(shape=(data.shape[0], 7))

    inputs[:, 0] = data[:, idx.sum] / data[:, idx.cols] / threads
    inputs[:, 1] = inputs[:, 0] / data[:, idx.max]
    inputs[:, 2] = (data[:, idx.sum] - data[:, idx.max]) / threads
    inputs[:, 3] = data[:, idx.rms] / threads
    inputs[:, 4] = (data[:, idx.sum] - data[:, idx.max]) / data[:, idx.cols] / threads
    inputs[:, 5] = (inputs[:, 0] - data[:, idx.cols]) / inputs[:, 0]
    inputs[:, 6] = 1. / data[:, idx.cols]

    # inputs[:, :idx.in_out_split] = data[:, :idx.in_out_split]
    # inputs[:, idx.avg] = data[:, idx.sum] / data[:, idx.cols]
    # inputs[:, idx.avg_without_max] = (data[:, idx.sum] - data[:, idx.max]) / data[:, idx.cols]
    # inputs[:, idx.sum_without_max] = data[:, idx.sum] - data[:, idx.max]
    # inputs[:, idx.sum_without_max + 1] = data[:, idx.rms] - data[:, idx.avg]
    # inputs[:, idx.sum_without_max + 2] = data[:, idx.cols] * data[:, idx.min]
    # inputs[:, idx.sum_without_max + 3] = data[:, idx.max] - data[:, idx.min]
    # inputs[:, idx.sum_without_max + 4] = data[:, idx.cols] / data[:, idx.avg]
    # inputs[:, idx.sum_without_max + 5] = data[:, idx.max] / data[:, idx.avg]

    iterations = data[:, idx.in_out_split:]
    outputs = np.argmin(iterations, axis=1)
    # (123123123, 16)
    return inputs, outputs, iterations

    # legacy data
    # return np.load("inputs.npy"), np.load("outputs.npy")


mask0 = [[1, 1, 1, 0, 1],
         [0, 0, 0, 1, 1],
         [0, 1, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 1, 1, 1],
         [0, 1, 1, 0, 0],
         [0, 1, 1, 0, 1]]

mask_without_min = [[1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 0],
                    [0, 1, 1, 0, 0],
                    [1, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1]]


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
    model.add(MaskedDense(5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(11, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def custom_loss(y_pred, y_true):
    return K.mean(tf.abs(tf.argmax(y_pred, axis=-1) - tf.argmax(y_true, axis=-1)) <= 1, axis=-1)


topk = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
topk.__name__ = "top3_acc"


def create_smaller(num_classes):
    model = Sequential()
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation="sigmoid"))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=["accuracy", custom_loss, topk])
    return model


def plot_result_hist(outputs):
    import matplotlib.pyplot as plt
    plt.hist(outputs)
    plt.show()


if __name__ == "__main__":
    # inputs, outputs, iterations = load_inputs_outputs(threads=1024, max_count=4500000)
    inputs, outputs, iterations = load_inputs_outputs(
        threads=64, max_count=1000000)
    num_classes = iterations.shape[1]
    print("num_classes", num_classes)
    print("samples", inputs.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.10, random_state=0)
    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

    values = X_train[:, -1]
    mean_val = np.average(values)

    batch_size = 128
    tb = TensorBoard(
        log_dir="logs/{}".format(datetime.now().strftime("%H:%M:%S")),
        update_freq=batch_size * 100
    )

    # snapshot = ModelCheckpoint(filepath="./test")

    model = create_smaller(num_classes)
    model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_test, y_test),
        callbacks=[tb],
        epochs=10,
        batch_size=batch_size,
    )

    model.save("{}_new.h5".format(1 << (iterations.shape[1] - 1)))
