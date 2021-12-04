import datetime
import time
from collections import namedtuple

import numpy as np

import torch
import torch.utils.data
from torch import nn
from torch import optim

from tsad import utils
from tsad.data import UCRTSAD2021Dataset, KPIDataset, YahooS5Dataset, TimeSeries
from tsad.model import AutoEncoder, RNNModel

PreparedData = namedtuple('PreparedData', field_names=["train", "valid", "test"])


class Train:

    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.dataset = self.get_data()
        self.model = None
        self.dataset_g = iter(self.dataset)
        self.criterion = self.get_loss()
        self.optimizer = None

        self.timestamp = "{:.0f}".format(datetime.datetime.now().timestamp())
        self.logger = utils.get_logger(self.config, f"{self.config.res}/{self.timestamp}.log")

    def get_model(self):
        if self.config.rnn_type in ["LSTM", "GRU"]:
            return RNNModel(self.config.history_w, self.config.predict_w, self.config.hidden_dim,
                            self.config.num_layers, self.config.dropout, self.config.rnn_type)
        else:
            raise ValueError(f"not support --rnn_type arg: {self.config.rnn_type}")

    def get_loss(self):
        if self.config.loss == "MAE":
            return nn.L1Loss(reduction="sum").to(self.device)
        elif self.config.loss == "MSE":
            return nn.MSELoss().to(self.device)
        else:
            raise ValueError(f"not support --loss arg: {self.config.loss}")

    def get_optim(self, model: nn.Module):
        if self.config.optim == "adam":
            return optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.optim == "adamW":
            return optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
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

    def run_once(self, file=None, prepared_data=None, prefix="A1Benchmark"):
        if file is not None:
            if self.config.dataset != "Yahoo":
                data_id, train, test, anomaly_vect = self.dataset.load_one(file)
            else:
                data_id, train, test, anomaly_vect = self.dataset.load_one(prefix, file)
        else:
            data_id, train, test, anomaly_vect = next(self.dataset_g)

        self.logger.info(f"Start training for {data_id} with config",
                         extra={"detail": f"\nlog_file: {self.timestamp}.log\n" + utils.dict2str(vars(self.config))
                                })

        if self.device.type == "cuda":
            self.logger.debug(utils.get_cuda_usage())

        if prepared_data is None:
            prepared_data = self.prepare_data(train, test, anomaly_vect)

        self.model = self.get_model().to(self.device)
        # self.model.init_weight()
        self.optimizer = self.get_optim(self.model)

        train_losses = []
        valid_losses = []
        best_valid_loss = None
        for epoch in range(self.config.epochs):
            start_time = time.time()
            losses = self.train(prepared_data.train, epoch)
            train_loss = np.mean(losses)
            valid_loss = self.valid(prepared_data.valid)
            self.logger.info("epoch {:03d}, time {:0<6.2f}s, train_loss {:0<6.3f}, valid loss {:0<6.3f}".format(
                epoch + 1, time.time() - start_time, train_loss, valid_loss))
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if best_valid_loss is None or valid_loss < best_valid_loss:
                with open(f"{self.config.res}/{data_id}_{self.config.output}", "wb") as f:
                    torch.save(self.model, f)
                best_valid_loss = valid_loss

        self.logger.info(f"Save model with valid loss: {best_valid_loss}")
        eval_loss = self.eval(*prepared_data.test, data_id=data_id)
        self.logger.info(f"eval loss on test set: {eval_loss}")

    def eval(self, test_data, label, data_id=None):
        with open(f"{self.config.res}/{data_id}_{self.config.output}", "rb") as f:
            self.model = torch.load(f, map_location=self.device)
        self.model.rnn.flatten_parameters()
        self.logger.info(f"Start eval on test set")
        self.model.eval()
        res_y = []
        res_y_ = []
        with torch.no_grad():
            for x_batch, y_batch in test_data:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                out = self.model(x_batch)
                res_y_.append(out.item())
                res_y.append(y_batch.item())

        return np.array(res_y_), np.array(res_y), label.dataset

    def valid(self, valid_data):
        self.model.eval()
        total_loss = 0
        total_batches = len(valid_data)
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(valid_data):
                loss = self.compute_loss(x_batch, y_batch)
                total_loss += loss.item()

        loss = total_loss / total_batches

        return loss

    def compute_loss(self, x_batch, y_batch):
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        out = self.model(x_batch)
        loss = self.criterion(out, y_batch)
        return loss

    def train(self, train_data, epoch):
        train_loss = []
        self.model.train()
        total_loss = 0
        total_batches = len(train_data)

        for batch_idx, (x_batch, y_batch) in enumerate(train_data):
            self.model.zero_grad()
            loss = self.compute_loss(x_batch, y_batch)
            loss.backward()
            if self.config.clip is not None:
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

    def prepare_data(self, train, test, anomaly_vect, test_batch_size=1):

        train_ts = TimeSeries(train, self.config.history_w,
                              pred_w=self.config.predict_w,
                              stride=self.config.stride, device=self.device)
        size = len(train_ts)
        valid_size = int(size * self.config.valid_prop)
        train_set, valid_set = torch.utils.data.random_split(train_ts, [size - valid_size, valid_size])

        test_set = TimeSeries(test, self.config.history_w,
                              pred_w=self.config.predict_w,
                              stride=self.config.stride, device=self.device)
        anomaly_vect = anomaly_vect[len(train):]

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config.batch_size,
                                                   drop_last=True, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=self.config.batch_size, drop_last=True)

        if test_batch_size is None:
            test_batch_size = len(test_set)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size)
        anomaly_vect_loader = torch.utils.data.DataLoader(anomaly_vect, batch_size=test_batch_size)
        return PreparedData(train=train_loader, valid=valid_loader, test=(test_loader, anomaly_vect_loader))

    def __iter__(self):
        self.dataset_g = iter(self.dataset)
        return self

    def __next__(self):
        return self.run_once()


class EncoderTrain(Train):

    def __init__(self, config):
        super(EncoderTrain, self).__init__(config)
        self.config.predict_w = 0

    def get_model(self):
        return AutoEncoder(self.config.history_w, self.config.emb_dim, self.config.hidden_dim, self.config.rnn_type)

    def compute_loss(self, x_batch, y_batch):
        out = self.model(x_batch)
        return self.criterion(out, x_batch)

    def eval(self, test_data, label, data_id=None):
        with open(f"{self.config.res}/{data_id}_{self.config.output}", "rb") as f:
            self.model = torch.load(f, map_location=self.device)
        self.logger.info(f"Start eval on test set")
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for x_batch, y_batch in test_data:
                loss = self.compute_loss(x_batch, y_batch)
                total_loss.append(loss.item())

        return np.mean(total_loss)

    def prepare_data(self, train, test, anomaly_vect, test_batch_size=None):
        return super(EncoderTrain, self).prepare_data(train, test, anomaly_vect, test_batch_size)
