import abc
import enum

import numpy as np
import torch
from scipy import optimize
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

    @abc.abstractmethod
    def fit(self, dataloader):
        pass

    @abc.abstractmethod
    def get_scores(self, dataloader):
        pass


class GaussianDetector(Detector):
    def __init__(self, model):
        super(GaussianDetector, self).__init__(model)
        self.u = None
        self.cov = None
        self.muvnormal = None
        self.model.eval()

    def fit(self, dataloader):
        err = []
        for batch_idx, (x_batch, y_batch) in dataloader:
            y_ = self.model(x_batch)
            err.append(functional.l1_loss(y_, y_batch, reduction="none"))
        err = torch.vstack(err).numpy()
        # [T, L, N]
        err = err.reshape(-1, err[-1])

        self.u = np.mean(err, axis=0)
        self.cov = np.cov(err, rowvar=False)
        self.muvnormal = multivariate_normal(self.u, self.cov, allow_singular=True)

    def get_scores(self, dataloader):
        for x, y in dataloader:
            seq_len, window_size = x.shape[0], x.shape[1]

            y_ = self.model(x)
            err = functional.l1_loss(y_, y, reduction="none").numpy()
            err = err.reshape(-1, err[-1])
            res_len = seq_len + window_size - 1
            if self.muvnormal is None:
                return np.zeros((res_len, 1))
            else:
                scores = -self.muvnormal.logpdf(err)
                scores = scores.reshape(-1, window_size)
                lattice = np.full((window_size, res_len), np.nan)
                for i, score in enumerate(scores):
                    lattice[i % window_size, i:i + window_size] = score

                scores = np.nanmean(lattice, axis=0)
                return scores


class Anomaly(enum.Enum):
    NORMAL = 0
    WEAK_ANOMALY = 1
    STRONG_ANOMALY = 2


class SPOTJudge:
    def __init__(self, inits, t0=None, q=0.01):
        self.t = t0
        self.peaks = list(inits[inits > t0] - t0)
        self.n = inits.shape[0]
        self.q = q
        self.zq = self.calc_threshold(*self.find_params())

    def is_anomaly(self, score):
        if score > self.zq:
            return Anomaly.STRONG_ANOMALY
        elif score > self.t:
            self.peaks.append(score - self.t)
            self.n += 1
            self.zq = self.calc_threshold(*self.find_params())
            return Anomaly.WEAK_ANOMALY
        else:
            self.n += 1
            return Anomaly.NORMAL

    def calc_threshold(self, gamma, sigma):
        n_t = len(self.peaks)
        return (sigma / gamma) * (np.power((self.q * self.n) / n_t, -gamma) - 1) + self.t

    @staticmethod
    def ux(x, peaks):
        res = peaks * x
        res = res + 1
        res = 1 / res
        return np.mean(res)

    @staticmethod
    def vx(x, peaks):
        res = peaks * x
        res = res + 1
        res = np.log(res)
        return np.mean(res) + 1

    @staticmethod
    def likelihood(gamma, sigma, peaks):
        n_t = peaks.shape[-1]
        peaks = peaks * gamma / sigma
        peaks = peaks + 1
        peaks = np.log(peaks)
        res = np.sum(peaks)
        res = res * (1 + 1 / gamma)
        res = -res - n_t * np.log(sigma)
        return res

    def get_bounds(self, epsilon=1e-6):
        max_ = np.max(self.peaks)
        min_ = np.min(self.peaks)
        mean = np.mean(self.peaks)
        bounds1 = [-1 / max_ + epsilon, epsilon]
        bounds2 = [2 * (mean - min_) / (mean * max_), 2 * (mean - min_) / (min_ * min_)]
        return bounds1, bounds2

    def find_params(self, num_x=8):
        bounds1, bounds2 = self.get_bounds()
        peaks = np.asarray(self.peaks, np.float64)
        f = lambda x: self.ux(x, peaks) * self.vx(x, peaks) - 1

        def target(x):
            x = np.asarray(x, np.float64)
            res = np.vectorize(f)(x)
            res = res ** 2
            res = np.sum(res)
            return res

        res1, res2 = None, None

        x0 = np.random.uniform(bounds1[0], bounds1[1], (num_x,))
        opt_res = optimize.minimize(target, method="L-BFGS-B", bounds=[bounds1] * num_x, x0=x0)
        if opt_res.success:
            res1 = opt_res.x

        x0 = np.random.uniform(bounds2[0], bounds2[1], (num_x,))
        opt_res = optimize.minimize(target, method="L-BFGS-B", bounds=[bounds2] * num_x, x0=x0)
        if opt_res.success:
            res2 = opt_res.x

        if res1 is not None and res2 is not None:
            res = np.hstack([res1, res2])
        elif res1 is None:
            res = res2
        else:
            res = res1

        def gamma_sigma(v):
            g = self.vx(v, peaks) - 1
            s = g / v
            return g, s

        max_likelihood = -np.inf
        gamma, sigmma = 0, 0
        for gi, si in map(gamma_sigma, res):
            p = self.likelihood(gi, si, peaks)
            if p > max_likelihood:
                max_likelihood = p
                gamma, sigmma = gi, si

        return gamma, sigmma


class EventDetector:

    @staticmethod
    def hamming_weight(n):
        count = 0
        while n > 0:
            n = n & (n - 1)
            count += 1

        return count

    @staticmethod
    def similarity(n1, n2):
        res = (EventDetector.hamming_weight(n1) * EventDetector.hamming_weight(n2)) ** 0.5
        return EventDetector.hamming_weight(n1 & n2) / res

    def __init__(self, size, pu=0.79, pl=0.11, th=0.5, a=0.5):
        self.size = size
        self.pu = pu
        self.pl = pl
        self.th = th
        self.a = a

        self.aset = {2 ** size - 1}

    @staticmethod
    def array2int(arr):
        arr = np.array(arr, dtype=np.int32)
        return int(f"0b{''.join(map(str, arr))}", 2)

    def is_anomaly(self, event):
        event_n = self.array2int(event)
        pe = self.hamming_weight(event_n) / self.size
        if pe > self.pu:
            self.aset.add(event_n)
            return True
        elif pe < self.pl:
            return False
        else:
            sim_max = 0
            for r in self.aset:
                sim = self.similarity(r, event_n)
                if sim > sim_max:
                    sim_max = sim

            if sim_max * self.a + (1 - self.a) * pe > self.th:
                self.aset.add(event_n)
                return True
            else:
                return False

