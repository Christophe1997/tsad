import abc
import os
import numpy as np
import pandas as pd
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


class CSVDataset(abc.ABC):

    def __init__(self, root_dir):
        root_dir = os.path.abspath(root_dir)
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"dataset not found in {root_dir}")

        self.root_dir = root_dir

    @abc.abstractmethod
    def __iter__(self):
        pass


class UCRTSAD2021Dataset(CSVDataset):

    def __init__(self, root_dir):
        super(UCRTSAD2021Dataset, self).__init__(root_dir)
        self.files = sorted(os.listdir(self.root_dir))

    def __iter__(self):
        for file in self.files:
            fullpath = os.path.join(self.root_dir, file)
            file_name = file.split(".")[0]
            idx, _, _, name, train_end, anomaly_start, anomay_end, = file_name.split("_")
            meta = {
                "index": idx,
                "name": name,
                "train_end": train_end,
                "anomaly_start": anomaly_start,
                "anomaly_end": anomay_end
            }
            data = pd.read_csv(fullpath).to_numpy()

            yield meta, data


class YahooS5Dataset(CSVDataset):
    def __init__(self, root_dir):
        super(YahooS5Dataset, self).__init__(root_dir)
        prefixs = os.listdir(self.root_dir)
        self.files = [(prefix, file) for prefix in os.listdir(self.root_dir)
                      for file in os.listdir(os.path.join(self.root_dir, prefix))]

    def __iter__(self):
        for prefix, file in self.files:
            full_path = os.path.join(self.root_dir, prefix, file)
            data = pd.read_csv(full_path, usecols=["value", "anomaly"])
            meta = {
                "prefix": prefix,
                "anomaly_vect": data["anomaly"].to_numpy(),
            }
            data = data["value"].to_numpy()
            yield meta, data


class KPIDataset(CSVDataset):
    def __init__(self, root_dir):
        super(KPIDataset, self).__init__(root_dir)
        