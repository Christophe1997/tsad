import abc
import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from tsad import utils


class SlidingWindowDataset(Dataset):

    def __init__(self, data: np.ndarray, history_w, pred_w=1, overlap=False, device=torch.device("cpu")):
        super(SlidingWindowDataset, self).__init__()
        self.history_w = history_w
        self.pred_w = pred_w
        data = np.squeeze(data)

        data = utils.scan(data, history_w + pred_w)
        if overlap:
            self.x, self.y = data[:-pred_w], data[pred_w:]
        else:
            self.x, self.y = np.hsplit(data, [self.history_w])

        self.x = torch.tensor(self.x).float().to(device)
        self.y = torch.tensor(self.y).float().to(device)

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

    def load_one(self, file):
        fullpath = os.path.join(self.root_dir, file)
        file_name = file.split(".")[0]
        idx, _, _, name, train_end, anomaly_start, anomaly_end = file_name.split("_")
        train_end = int(train_end)
        anomaly_start = int(anomaly_start)
        anomaly_end = int(anomaly_end)

        data_id = f"ucr_{idx}_{name}"
        data = pd.read_csv(fullpath).to_numpy()
        anomaly_vect = np.zeros(len(data))
        anomaly_vect[anomaly_start - 1: anomaly_end] = 1
        indices = [train_end, len(data)]
        train, test, _ = np.split(data, indices)

        return data_id, utils.normalized(train), utils.normalized(test), anomaly_vect

    def __iter__(self):
        for file in self.files:
            yield self.load_one(file)


class YahooS5Dataset(CSVDataset):
    def __init__(self, root_dir, test_prop=0.3):
        super(YahooS5Dataset, self).__init__(root_dir)
        self.files = [(prefix, file) for prefix in os.listdir(self.root_dir) if prefix.endswith("Benchmark")
                      for file in os.listdir(os.path.join(self.root_dir, prefix))]
        self.train_prop = 1 - test_prop

    def load_one(self, prefix, file):
        full_path = os.path.join(self.root_dir, prefix, file)
        try:
            data = pd.read_csv(full_path, usecols=["value", "anomaly"])
        except ValueError:
            data = pd.read_csv(full_path, usecols=["value", "is_anomaly"])

        data.columns = ["value", "label"]
        data_id = f"yahoo_{prefix}_{file.split('.')[0]}"
        anomaly_vect = data["label"].to_numpy()
        data = data["value"].to_numpy()
        indices = [math.floor(len(data) * self.train_prop), len(data)]
        train, test, _ = np.split(data, indices)

        return data_id, utils.normalized(train), utils.normalized(test), anomaly_vect

    def __iter__(self):
        for prefix, file in self.files:
            yield self.load_one(prefix, file)


class KPIDataset(CSVDataset):
    def __init__(self, root_dir, train="phase2_train.csv", test="phase2_test.csv"):
        super(KPIDataset, self).__init__(root_dir)
        self.train_data = pd.read_csv(os.path.join(self.root_dir, train), usecols=["value", "label", "KPI ID"])
        self.test_data = pd.read_csv(os.path.join(self.root_dir, test), usecols=["value", "label", "KPI ID"])

    def load_one(self, kpi_id):
        train_df = self.train_data.loc[self.train_data["KPI ID"] == kpi_id][["value", "label"]]
        test_df = self.test_data.loc[self.test_data["KPI ID"] == kpi_id][["value", "label"]]
        data_id = f"kpi_{kpi_id}"
        train = train_df["value"].to_numpy()
        test = test_df["value"].to_numpy()
        anomaly_vect = np.hstack((train_df["label"].to_numpy(), test_df["label"].to_numpy()))
        return data_id, utils.normalized(train), utils.normalized(test), anomaly_vect

    def __iter__(self):
        for kpi_id in self.train_data["KPI ID"].unique():
            yield self.load_one(kpi_id)


class PreparedData:

    def __init__(self, train: np.ndarray, test: np.ndarray, anomaly_vect: np.ndarray, valid_prop=0.3):
        train = train.squeeze()
        size = train.shape[0]
        train_size = math.floor(size * (1 - valid_prop))
        self.train = train[:train_size]
        self.valid = train[train_size + 1:]
        self.test = test.squeeze()
        anomaly_vect = anomaly_vect.squeeze()

        self.train_size = train_size
        self.valid_size = size - train_size
        self.test_size = self.test.shape[0]

        assert anomaly_vect.shape[0] == self.train_size + self.valid_size + self.test_size, "anomaly size not match"

        self.train_anomaly = anomaly_vect[:train_size]
        self.valid_anomaly = anomaly_vect[train_size + 1: size]
        self.test_anomaly = anomaly_vect[-self.test_size:]

    def batchify(self, history_w, pred_w, batch_size,
                 overlap=False,
                 shuffle=True,
                 test_batch_size=None,
                 device=torch.device("cpu")):
        if test_batch_size is None:
            test_batch_size = batch_size

        train_loader = DataLoader(SlidingWindowDataset(self.train, history_w, pred_w, overlap, device=device),
                                  batch_size=batch_size, shuffle=shuffle)

        valid_loader = DataLoader(SlidingWindowDataset(self.valid, history_w, pred_w, overlap, device=device),
                                  batch_size=batch_size, shuffle=False)

        test_loader = DataLoader(SlidingWindowDataset(self.test, history_w, pred_w, overlap, device=device),
                                 batch_size=test_batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader
