import pickle
import numpy as np


def save_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        print(" [*] save %s" % path)


def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        print(" [*] load %s" % path)
        return obj


def save_npy(path, obj):
    np.save(path, obj)
    print(" [*] save %s" % path)


def load_npy(path):
    obj = np.load(path)
    print(" [*] load %s" % path)
    return obj


def get_position_encoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P
