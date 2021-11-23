import argparse
import logging
import time
from collections import namedtuple

import torch
from torch import nn
from torch import optim
import torch.utils.data

from tsad import utils
from tsad.data import UCRTSAD2021Dataset, KPIDataset, YahooS5Dataset, TimeSeries
from tsad.model import RNNModel

PreparedData = namedtuple('PreparedData', field_names=["train", "valid", "test", "test_one"])

parser = argparse.ArgumentParser("Train Script")
parser.add_argument("--data", type=str, default="./data/UCR_TimeSeriesAnomalyDatasets2021/UCR_Anomaly_FullData",
                    help="root dir of data")
parser.add_argument("--dataset", type=str, default="UCR", help="dataset type(UCR, KPI, Yahoo)")
parser.add_argument("--model", type=str, default="LSTM", help="model type")
parser.add_argument("--verbose", dest="log_level", action="store_const", const=logging.DEBUG,
                    help="debug logging")
parser.set_defaults(log_level=logging.INFO)
parser.add_argument("-O", "--output", type=str, default="model.pt", help="model save path")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--rnn_type", type=str, default="LSTM",
                    help="RNN type used for train(LSTM, GRU)")
parser.add_argument("--hidden_dim", type=int, default=256,
                    help="number of hidden units per layer")
parser.add_argument("--num_layers", type=int, default=1,
                    help="number of hidden layers")
parser.add_argument("--history_w", type=int, default=45,
                    help="history window size for predicting")
parser.add_argument("--predict_w", type=int, default=1,
                    help="predict window size")
parser.add_argument("--stride", type=int, default=1,
                    help="stride for sliding window")
parser.add_argument("--dropout", type=int, default=0.5,
                    help="dropout applied to layers")
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--epochs", type=int, default=50, help="epoch limit")
parser.add_argument("--valid_prop", type=float, default=0.2, help="validation set prop")
parser.add_argument("--test_prop", type=float, default=0.3, help="test set prop(only work for some dataset)")
parser.add_argument("--loss", type=str, default="L1", help="loss function")
parser.add_argument("--optim", type=str, default="adam", help="optimizer")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
parser.add_argument("--per_batch", type=int, default=5, help="log frequence")

args = parser.parse_args()


class Train:

    def __init__(self, config):
        self.config = config
        self.model = self.get_model()
        self.dataset = iter(self.get_data())
        self.criterion = self.get_loss()
        self.optimizer = self.get_optim(self.model)
        self.logger = utils.get_logger(config)

    def get_model(self):
        if self.config.rnn_type in ["LSTM", "GRU"]:
            return RNNModel(self.config.history_w, self.config.predict_w, self.config.hidden_dim,
                            self.config.num_layers, self.config.dropout, self.config.rnn_type)
        else:
            raise ValueError(f"not support --rnn_type arg: {self.config.rnn_type}")

    def get_loss(self):
        if self.config.loss == "L1":
            return nn.L1Loss()
        else:
            raise ValueError(f"not support --loss arg: {self.config.loss}")

    def get_optim(self, model: nn.Module):
        if self.config.optim == "adam":
            return optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"not support --optim arg: {self.config.optim}")

    def get_data(self):
        if self.config.dataset == "UCR":
            return UCRTSAD2021Dataset(self.config.data)
        elif self.config.dataset == "KPI":
            return KPIDataset(self.config.data)
        elif self.config.dataset == "Yahoo":
            return YahooS5Dataset(self.config.data, self.config.test_prop)
        else:
            raise ValueError(f"not support --dataset arg: {self.config.dataset}")

    def run_once(self):
        try:
            data_id, train, test, anomaly_vect = next(self.dataset)
        except StopIteration:
            self.logger.warning("Dataset ex/hausted.")
            return False

        self.logger.info(f"Start training for {data_id} with config",
                         extra={"detail": utils.dict2str(vars(self.config))})

        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            self.logger.debug(utils.get_cuda_usage())

        prepared_data = self.prepare_data(train, test, anomaly_vect)

        train = prepared_data.train
        for epoch in range(self.config.epoch):
            self.model.train()
            total_loss = 0
            total_batches = len(train)
            start_time = time.time()
            hidden = self.model.init_hidden(self.config.batch_size)
            for batch_idx, (x_batch, y_batch) in enumerate(train):
                self.model.zero_grad()
                hidden = utils.repackage_hidden(hidden)
                x_batch = x_batch.view(-1, self.config.batch_size, self.config.history_w).to(device)
                y_batch = y_batch.to(device)
                out, hidden = self.model(x_batch, hidden)
                loss = self.criterion(out, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()

                total_loss += loss.item()
                if batch_idx > 0 and batch_idx % self.config.per_batch == 0:
                    cur_loss = total_loss / self.config.per_batch
                    self.logger.info("epoch {:3d}, {}/{} batches, loss {:.2f}".format(
                        epoch + 1, batch_idx + 1, total_batches, cur_loss))
                    total_loss = 0

            self.model.eval()

    def prepare_data(self, train, test, anomaly_vect):

        train_ts = TimeSeries(train, self.config.history_w,
                              pred_w=self.config.pred_w,
                              stride=self.config.stride)
        size = len(train_ts)
        valid_size = int(size * self.config.valid_prop)
        train_set, valid_set = torch.utils.data.random_split(train, [size - valid_size, valid_size])

        test_set = TimeSeries(test, self.config.history_w,
                              pred_w=self.config.pred_w,
                              stride=self.config.stride)
        anomaly_vect = anomaly_vect[len(train) + self.config.stride:]
        assert len(test_set) == len(anomaly_vect)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config.batch_size,
                                                   drop_last=True, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=self.config.batch_size, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.config.batch_size, drop_last=True)
        anomaly_vect_loader = torch.utils.data.DataLoader(anomaly_vect, batch_size=self.config.batch_size,
                                                          drop_last=True)

        test_loader_one = torch.utils.data.DataLoader(test_set, batch_size=len(test_set))
        anomaly_vect_loader_one = torch.utils.data.DataLoader(anomaly_vect, batch_size=len(test_set))
        res = PreparedData(train=train_loader, valid=valid_loader, test=(test_loader, anomaly_vect_loader),
                           test_one=(test_loader_one, anomaly_vect_loader_one))

        return res
