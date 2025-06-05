import numpy as np
import time

def load_data(filename):
    data = np.loadtxt(filename)
    return data[:, 1:], data[:, 0]

def nearest_neighbor(train_data, test_instance, features):
    distances = np.sqrt(np.sum((train_data[:, features] - test_instance[features])**2, axis=1))
    return np.argmin(distances)


