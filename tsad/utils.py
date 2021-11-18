import numpy as np


def scan1d(arr: np.ndarray, window_size, stride=1):
    size = len(arr)
    if window_size > size:
        raise ValueError("window size must smaller than arr length")

    shape = (size - window_size) // stride + 1, window_size
    elem_size = arr.strides[-1]
    return np.lib.stride_tricks.as_strided(arr, shape, strides=(elem_size * stride, elem_size), writeable=False)
