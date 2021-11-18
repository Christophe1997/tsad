import numpy as np
from torch.utils.data import Dataset

from tsad import utils


class TimeSeries(Dataset):

    def __init__(self, data: np.ndarray, history_w, pred_w=1, stride=1, transform=None):
        super(TimeSeries, self).__init__()
        self.history_w = history_w
        self.pred_w = pred_w
        self.x = None
        self.y = None
        data = np.squeeze(data)
        if len(data.shape) != 1:
            raise ValueError("Only support 1D data")

        data = utils.scan1d(data, history_w + pred_w, stride=stride)
        self.x, self.y = np.hsplit(self.data, [self.history_w])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
