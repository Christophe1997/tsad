import abc

import numpy as np
import torch
import torch.nn.functional


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


class Detector:

    @abc.abstractmethod
    def detect(self, observed):
        pass


class AutoEncoderDetector(Detector):

    def __init__(self, model, predict_window, threshold=3):
        self.model = model.to(torch.device("cpu"))
        self.predict_window = predict_window
        self.threshold = threshold

        self.queue = FixSizedQueue(predict_window)
        self.peaks = []
        self.n = predict_window - 1

    def detect(self, observed):
        self.queue.enqueue(observed)
        if not self.queue.is_full():
            return observed, False
        else:
            actual = torch.tensor(self.queue.to_numpy()).float()
            self.model.eval()
            with torch.no_grad():
                pred = self.model(actual)
            pred_val = pred[:, -1].item()
            return pred_val, self.is_anomaly(pred)

    def is_anomaly(self, pred):
        actual = torch.tensor(self.queue.to_numpy()).float()
        score = torch.nn.functional.mse_loss(pred, actual)
        return score > self.threshold

    # Attempt to apply extreme value theory failed
    #
    # def is_anomaly(self, pred_val, observed):
    #     self.n += 1
    #     residue = np.abs(pred_val - observed)
    #     if residue < self.threshold:
    #         return False
    #     else:
    #         self.update_threshold(residue)
    #         return True
    #
    # def update_threshold(self, residue):
    #     self.peaks.append(residue)
    #     if len(self.peaks) < 2:
    #         return
    #     else:
    #         roots = self.find_root()
    #         if len(roots) == 0:
    #             return
    #         else:
    #             gamma, sigma = self.find_best(roots)
    #             n_t = len(self.peaks)
    #             q = 0.01
    #             self.threshold = (sigma / gamma) * (np.power((q * self.n) / n_t, -gamma) - 1)
    #
    # def find_best(self, roots):
    #     gamma1 = self.vx(roots[0]) - 1
    #     sigma1 = gamma1 / roots[0]
    #     if len(roots) == 1:
    #         return gamma1, sigma1
    #     else:
    #         gamma2 = self.vx(roots[1]) - 1
    #         sigma2 = gamma2 / roots[1]
    #         if self.likelihood(gamma1, sigma1) > self.likelihood(gamma2, sigma2):
    #             return gamma1, sigma1
    #         else:
    #             return gamma2, sigma2
    #
    # def likelihood(self, gamma, sigma):
    #     peaks = np.array(self.peaks)
    #     n_t = peaks.shape[-1]
    #     peaks = peaks * gamma / sigma
    #     peaks = peaks + 1
    #     peaks = np.log(peaks)
    #     res = np.sum(peaks)
    #     res = res * (1 + 1 / gamma)
    #     res = -res - n_t * np.log(sigma)
    #     return res
    #
    # def find_root(self):
    #     f = lambda x: self.ux(x) * self.vx(x) - 1
    #     bounds1, bounds2 = self.get_bounds()
    #     roots = []
    #     if f(bounds1[0]) * f(bounds1[1]) < 0:
    #         roots.append(optimize.root_scalar(f, method="brentq", bracket=bounds1).root)
    #     if f(bounds2[0]) * f(bounds2[1]) < 0:
    #         roots.append(optimize.root_scalar(f, method="brentq", bracket=bounds2).root)
    #
    #     return roots
    #
    # def ux(self, x):
    #     peaks = np.array(self.peaks)
    #     res = peaks * x
    #     res = res + 1
    #     res = 1 / res
    #     return np.mean(res)
    #
    # def vx(self, x):
    #     peaks = np.array(self.peaks)
    #     res = peaks * x
    #     res = res + 1
    #     res = np.log(res)
    #     return np.mean(res) + 1
    #
    # def get_bounds(self):
    #     max_ = np.max(self.peaks)
    #     min_ = np.min(self.peaks)
    #     bounds1 = [-1 / max_ + 0.01, -0.01]
    #     bounds2 = [0.01, 2 * (np.mean(self.peaks) - min_) / (min_ * min_) - 0.01]
    #     return bounds1, bounds2
