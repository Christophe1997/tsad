import datetime
import logging
import os
import sys

import numpy as np
import torch


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
    fp_handler = logging.FileHandler(fp, mode='a+', encoding='utf8')
    s_handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s] %(message)s\n%(detail)s')
    fp_handler.setFormatter(formatter)
    s_handler.setFormatter(formatter)
    res.addHandler(fp_handler)
    res.addHandler(s_handler)
    res.setLevel(level=args.log_level)
    return res


def create_dir(full_path):
    abs_path = os.path.abspath(full_path)
    dir_path = os.path.dirname(abs_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def dict2str(d: dict):
    return "\n".join(f"{k}: {v}" for k, v in d.items())


def get_cuda_usage():
    if not torch.cuda.is_available():
        return "Not available"
    else:
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        resolved_mem = torch.cuda.memory_reserved(device_id) / 1024 ** 3
        allocated_mem = torch.cuda.memory_allocated(device_id) / 1024 ** 3
        return "{}: {:.2f}GB/{:.2f}GB({:.2f}%)".format(device_name,
                                                       allocated_mem,
                                                       resolved_mem,
                                                       allocated_mem / resolved_mem * 100)


def repackage_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(v) for v in hidden)
