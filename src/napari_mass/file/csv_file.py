import csv
import numpy as np
import os
import pandas as pd


def csv_read(filename, delimiter=','):
    data = None
    if os.path.exists(filename):
        data = pd.read_csv(filename, header=None, delimiter=delimiter)
        return data.to_numpy()
    return data


def csv_read0(filename, delimiter=','):
    data = None
    if os.path.exists(filename):
        with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            data = [row for row in reader]
    return data


def csv_write(filename, data, append=False, delimiter=','):
    mode = 'a' if append else 'w'
    if isinstance(data, np.ndarray) and data.ndim > 1:
        data = data.tolist()
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerows(data)
