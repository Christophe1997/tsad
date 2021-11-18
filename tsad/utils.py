import datetime
import logging
import os

import numpy as np


def scan1d(arr: np.ndarray, window_size, stride=1):
    size = len(arr)
    if window_size > size:
        raise ValueError("window size must smaller than arr length")

    shape = (size - window_size) // stride + 1, window_size
    elem_size = arr.strides[-1]
    return np.lib.stride_tricks.as_strided(arr, shape, strides=(elem_size * stride, elem_size), writeable=False)


def get_logger(args):
    res = logging.getLogger(__name__)
    fp = "{:.0f}.log".format(datetime.datetime.now().timestamp())
    handler = logging.FileHandler(fp, mode='a+', encoding='utf8')
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s] %(message)s')
    handler.setFormatter(formatter)
    res.addHandler(handler)
    res.setLevel(level=args.log_level)
    return res


def create_dir(full_path):
    abs_path = os.path.abspath(full_path)
    dir_path = os.path.dirname(abs_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path
