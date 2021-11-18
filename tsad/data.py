from torch.utils.data import Dataset
import numpy as np
import utils


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

        self.data = utils.scan1d(data, history_w + pred_w, stride=stride)

    def __getitem__(self, index):
        return self.data[index, :self.history_w], self.data[index, self.history_w + 1:]

    def x_and_y(self):
        if self.x is None:
            self.x, self.y = np.hsplit(self.data, [self.history_w])
        return self.x, self.y
