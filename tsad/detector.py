import abc
import logging

import numpy as np
import torch
from scipy.stats import multivariate_normal
from torch.nn import functional


class FixSizedQueue:

    def __init__(self, size):
        self.capacity = size
        self.__data = []
        self.size = 0

    def is_full(self):
        return self.size == self.capacity

    def enqueue(self, val):
        self.__data.append(val)
        self.size += 1
        if self.size > self.capacity:
            self.__data.pop(0)
            self.size -= 1

    def dequeue(self):
        self.size -= 1
        return self.__data.pop(0)

    def to_numpy(self):
        return np.array(self.__data)[np.newaxis, :]


class Detector(abc.ABC):

    def __init__(self, model):
        self.model = model
        self.logger = logging.getLogger("root")

    @abc.abstractmethod
    def fit(self, dataloader):
        pass

    @abc.abstractmethod
    def detect(self, x, y):
        pass


class GaussianDetector(Detector):
    def __init__(self, model):
        super(GaussianDetector, self).__init__(model)
        self.u = None
        self.cov = None
        self.muvnormal = None

    def fit(self, dataloader):
        err = []
        for batch_idx, (x_batch, y_batch) in dataloader:
            y_ = self.model(x_batch)
            err.append(functional.l1_loss(y_, y_batch, reduction="none"))
        err = torch.vstack(err).numpy()
        # [T, L, N]
        err = err.reshape(-1, err[-1])
        self.logger.debug(f"During fitting, the error vector shape is {err.shape}")

        self.u = np.mean(err, axis=0)
        self.cov = np.cov(err, rowvar=False)
        self.muvnormal = multivariate_normal(self.u, self.cov, allow_singular=True)

    def detect(self, x, y):
        seq_len, window_size = x.shape[0], x.shape[1]

        y_ = self.model(x)
        err = functional.l1_loss(y_, y, reduction="none").numpy()
        err = err.reshape(-1, err[-1])
        res_len = seq_len + window_size - 1
        self.logger.debug(f"During detecting, the error vector shape is {err.shape}")
        if self.muvnormal is None:
            self.logger.warning("Detecting before fit")
            return np.zeros((res_len, 1))
        else:
            scores = -self.muvnormal.logpdf(err)
            scores = scores.reshape(-1, window_size)
            lattice = np.full((window_size, res_len), np.nan)
            for i, score in enumerate(scores):
                lattice[i % window_size, i:i + window_size] = score

            scores = np.nanmean(lattice, axis=0)
            return scores
