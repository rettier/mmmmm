from keras.engine.saving import load_model
import numpy as np
from openpyxl import Workbook

input_keys = ['max', 'min', 'sum', 'sum_without_max', 'avg', 'avg_without_max', 'rows']


def to_igschel(model, name):
    wb = Workbook()
    ws = wb.active

    input_data = [1, 1, 1, 1, 1, 1, 1]

    for i, x in enumerate(input_data):
        ws.cell(row=i + 1, column=2).value = input_keys[i]
        ws.cell(row=i + 1, column=2).value = x

    for x in range(5):
        ws.cell(row=x + 1, column=4).value = "SUM("

    w = model.get_weights()
    for y in range(4):
        weights = wb.create_sheet("weights{}".format(y))
        wy = w[y]
        shape = wy.shape
        if len(shape) == 2:
            for ix, iy in np.ndindex(shape):
                weights.cell(row=ix + 1, column=iy + 1).value = float(wy[ix, iy])
        else:
            for ix in range(shape[0]):
                weights.cell(row=ix + 1, column=1).value = float(wy[ix])

    wb.save(name + ".xlsx")


to_igschel(load_model("models/7x5x11-full.h5"), "test")
