import numpy as np

def load_data(filename):
    data = np.loadtxt(filename)
    return data[:, 1:], data[:, 0]

def forward_selection(features, classes):

