import argparse
import logging
import datetime
import os
import time
from collections import namedtuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch import nn
from torch import optim
import torch.utils.data

from tsad import utils
from tsad.data import UCRTSAD2021Dataset, KPIDataset, YahooS5Dataset, TimeSeries
from tsad.model import RNNModel

PreparedData = namedtuple('PreparedData', field_names=["train", "valid", "test"])
Measure = namedtuple('Measure', field_names=["precision", "recall", "f1_score"])

parser = argparse.ArgumentParser("Train Script")
parser.add_argument("--data", type=str, default="./data/UCR_TimeSeriesAnomalyDatasets2021/UCR_Anomaly_FullData",
                    help="root dir of data")
parser.add_argument("--dataset", type=str, default="UCR",
                    help="dataset type(UCR, KPI, Yahoo), default 'UCR'")
parser.add_argument("--verbose", dest="log_level", action="store_const", const=logging.DEBUG,
                    help="debug logging")
parser.set_defaults(log_level=logging.INFO)
parser.add_argument("-O", "--output", type=str, default="model.pt", help="model save path, default 'model.pt'")
parser.add_argument("--seed", type=int, default=1234, help="random seed, default 1234")
parser.add_argument("--rnn_type", type=str, default="LSTM",
                    help="RNN type used for train(LSTM, GRU), default 'LSTM'")
parser.add_argument("--hidden_dim", type=int, default=128,
                    help="number of hidden units per layer, default 128")
parser.add_argument("--num_layers", type=int, default=2,
                    help="number of hidden layers, default 2")
parser.add_argument("--history_w", type=int, default=32,
                    help="history window size for predicting, default 32")
parser.add_argument("--predict_w", type=int, default=1,
                    help="predict window size, default 1")
parser.add_argument("--stride", type=int, default=1,
                    help="stride for sliding window, default 1")
parser.add_argument("--dropout", type=float, default=0.2,
                    help="dropout applied to layers, default 0.2")
parser.add_argument("--batch_size", type=int, default=20, help="batch size, default 20")
parser.add_argument("--epochs", type=int, default=50, help="epoch limit, default 50")
parser.add_argument("--valid_prop", type=float, default=0.2, help="validation set prop, default 0.2")
parser.add_argument("--test_prop", type=float, default=0.3,
                    help="test set prop(only work for some dataset), default 0.3")
parser.add_argument("--loss", type=str, default="L1", help="loss function, default 'L1'")
parser.add_argument("--optim", type=str, default="adam", help="optimizer, default 'adam'")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate, default 1e-3")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay, default 1e-4")
parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping, default 0.25")
parser.add_argument("--per_batch", type=int, default=10, help="log frequence per batch, default 10")
parser.add_argument("--res", type=str, default="out", help="log files and result files dir")
parser.add_argument("--one_file", type=str, help="only train on the specific data")
parser.set_defaults(one_file=None)

args = parser.parse_args()


class Train:

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.get_model().to(self.device)
        self.dataset = self.get_data()
        self.dataset_g = iter(self.dataset)
        self.criterion = self.get_loss()
        self.optimizer = self.get_optim(self.model)

        self.timestamp = "{:.0f}".format(datetime.datetime.now().timestamp())
        self.logger = utils.get_logger(self.config, f"{self.config.res}/{self.timestamp}.log")
        self.delta = None

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

    def run_once(self, file=None):
        if file is not None:
            if self.config.dataset != "Yahoo":
                data_id, train, test, anomaly_vect = self.dataset.load_one(file)
            else:
                data_id, train, test, anomaly_vect = self.dataset.load_one("A1Benchmark", file)
        else:
            data_id, train, test, anomaly_vect = next(self.dataset_g)

        self.logger.info(f"Start training for {data_id} with config",
                         extra={"detail": f"\nlog_file: {self.timestamp}.log\n" + utils.dict2str(vars(self.config))
                                })

        if self.device.type == "cuda":
            self.logger.debug(utils.get_cuda_usage())

        prepared_data = self.prepare_data(train, test, anomaly_vect)

        self.model.init_weight()
        train_losses = []
        valid_losses = []
        for epoch in range(self.config.epochs):
            start_time = time.time()
            losses = self.train(prepared_data.train, epoch)
            train_loss = np.mean(losses)
            valid_loss = self.valid(prepared_data.valid)
            self.logger.info("epoch {:03d}, time {:0<6.2f}s, train_loss {:0<6.3f}, valid loss {:0<6.3f}".format(
                epoch + 1, time.time() - start_time, train_loss, valid_loss))
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

        res_y_, res_y, label, m = self.eval(*prepared_data.test)
        self.logger.info("Measure on test set: precision:"
                         " {:0<5.2f}, recall: {:0<5.2f}, F1 score: {:0<5.2f}".format(*m))

        return data_id, train_losses, valid_losses, res_y_, res_y, label, m

    def eval(self, test_data, label, delta=None):
        if delta is None:
            delta = self.delta
        self.logger.info(f"eval on test set with threshold: {delta}")
        self.model.eval()
        res_y = []
        res_y_ = []
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        label_g = iter(label)
        hidden = self.model.init_hidden(self.config.batch_size)
        with torch.no_grad():
            for x_batch, y_batch in test_data:
                hidden = utils.repackage_hidden(hidden)
                x_batch = x_batch.view(-1, self.config.batch_size, self.config.history_w).to(self.device)
                out, hidden = self.model(x_batch, hidden)
                res_y_.append(out.squeeze())
                res_y.append(y_batch.squeeze())
                is_pos = np.array(np.abs(res_y_[-1] - res_y[-1]) > delta)
                label_batch = np.array(next(label_g) == 1)
                for i, val in enumerate(label_batch):
                    if is_pos[i] and val:
                        tp += 1
                    elif is_pos[i]:
                        fp += 1
                    elif val:
                        fn += 1
                    else:
                        tn += 1

        self.logger.info(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        return np.hstack(res_y_), np.hstack(res_y), label.dataset, Measure(precision, recall, f1_score)

    def valid(self, valid_data):
        self.model.eval()
        total_loss = 0
        total_batches = len(valid_data)
        hidden = self.model.init_hidden(self.config.batch_size)
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(valid_data):
                hidden = utils.repackage_hidden(hidden)
                x_batch = x_batch.view(-1, self.config.batch_size, self.config.history_w).to(self.device)
                y_batch = y_batch.to(self.device)
                out, hidden = self.model(x_batch, hidden)
                total_loss += self.criterion(out, y_batch).item()

        return total_loss / total_batches

    def train(self, train_data, epoch):
        train_loss = []
        self.model.train()
        total_loss = 0
        total_batches = len(train_data)
        hidden = self.model.init_hidden(self.config.batch_size)

        for batch_idx, (x_batch, y_batch) in enumerate(train_data):
            self.model.zero_grad()
            hidden = utils.repackage_hidden(hidden)
            x_batch = x_batch.view(-1, self.config.batch_size, self.config.history_w).to(self.device)
            y_batch = y_batch.to(self.device)
            out, hidden = self.model(x_batch, hidden)
            loss = self.criterion(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
            self.optimizer.step()

            total_loss += loss.item()
            if batch_idx > 0 and (batch_idx + 1) % self.config.per_batch == 0:
                cur_loss = total_loss / self.config.per_batch
                train_loss.append(cur_loss)
                self.logger.debug("epoch {:03d}, {:05d}/{:05d} batches, loss {:0<6.3f}".format(
                    epoch + 1, batch_idx + 1, total_batches, cur_loss))
                total_loss = 0

        return train_loss

    def prepare_data(self, train, test, anomaly_vect):

        self.delta = np.std(train)
        train_ts = TimeSeries(train, self.config.history_w,
                              pred_w=self.config.predict_w,
                              stride=self.config.stride)
        size = len(train_ts)
        valid_size = int(size * self.config.valid_prop)
        train_set, valid_set = torch.utils.data.random_split(train_ts, [size - valid_size, valid_size])

        test_set = TimeSeries(test, self.config.history_w,
                              pred_w=self.config.predict_w,
                              stride=self.config.stride)
        anomaly_vect = anomaly_vect[len(train) + self.config.history_w:]
        assert len(test_set) == len(anomaly_vect), f"{len(test_set)} != {len(anomaly_vect)}"

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config.batch_size,
                                                   drop_last=True, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=self.config.batch_size, drop_last=True)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.config.batch_size, drop_last=True)
        anomaly_vect_loader = torch.utils.data.DataLoader(anomaly_vect, batch_size=self.config.batch_size,
                                                          drop_last=True)
        return PreparedData(train=train_loader, valid=valid_loader, test=(test_loader, anomaly_vect_loader))

    def __iter__(self):
        self.dataset_g = iter(self.dataset)
        return self

    def __next__(self):
        return self.run_once()

    def stats(self, name, loss_train, loss_valid, y_, y, label, measure):
        (indexes,) = np.where(label == 1)
        intervals = utils.get_intevals(indexes)
        fig = make_subplots(rows=2, cols=1, subplot_titles=["loss curve", "fit curve"])
        fig.add_trace(go.Scatter(x=np.arange(len(loss_train)), y=loss_train, mode="lines", name="train loss"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(len(loss_valid)), y=loss_valid, mode="lines", name="valid loss"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(len(y_)), y=y_, mode="lines", name="prediction",
                                 line={"color": "red"}), row=2, col=1)
        fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y, mode="lines", name="fact",
                                 line={"color": "green"}), row=2, col=1)
        min_y = np.min(y)
        max_y = np.max(y)
        for idx in intervals:
            fig.add_shape(go.layout.Shape(type="line", xref='x', x0=idx, y0=min_y, x1=idx, y1=max_y,
                                          line={"dash": "dash", "color": "blue"}), row=2, col=1)

        fig.update_layout(title="precision: {:0<5.2f}, recall: {:0<5.2f}, F1 score: {:0<5.2f}".format(*measure))

        fig.write_html(f"./{self.config.res}/{name}_{self.timestamp}.html")


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

os.makedirs(args.res, exist_ok=True)
main = Train(args)

if main.config.one_file is not None:
    res = main.run_once(main.config.one_file)
    main.stats(*res)
else:
    for res in main:
        main.stats(*res)
