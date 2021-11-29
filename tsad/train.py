import datetime
import time
from collections import namedtuple

import numpy as np
import plotly.graph_objects as go
import sklearn.metrics as metrics
import torch
import torch.utils.data
from plotly.subplots import make_subplots
from torch import nn
from torch import optim

from tsad import utils
from tsad.data import UCRTSAD2021Dataset, KPIDataset, YahooS5Dataset, TimeSeries
from tsad.model import RNNModel

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
            return nn.L1Loss()
        elif self.config.loss == "MSE":
            return nn.MSELoss()
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
        self.model.init_weight()
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
        res_y_, res_y, label = self.eval(*prepared_data.test, data_id=data_id)

        return data_id, train_losses, valid_losses, res_y_, res_y, label

    def eval(self, test_data, label, data_id=None):
        with open(f"{self.config.res}/{data_id}_{self.config.output}", "rb") as f:
            self.model = torch.load(f, map_location=self.device)
        self.model.rnn.flatten_parameters()
        self.logger.info(f"Start eval on test set")
        self.model.eval()
        res_y = []
        res_y_ = []
        hidden = self.model.init_hidden(1)
        with torch.no_grad():
            for x_batch, y_batch in test_data:
                hidden = utils.repackage_hidden(hidden)
                x_batch = x_batch.view(-1, 1, self.config.history_w).to(self.device)
                y_batch = y_batch.to(self.device)
                out, hidden = self.model(x_batch, hidden)
                res_y_.append(out.item())
                res_y.append(y_batch.item())

        return np.array(res_y_), np.array(res_y), label.dataset

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

        loss = total_loss / total_batches

        return loss

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

    def prepare_data(self, train, test, anomaly_vect):

        train_ts = TimeSeries(train, self.config.history_w,
                              pred_w=self.config.predict_w,
                              stride=self.config.stride, device=self.device)
        size = len(train_ts)
        valid_size = int(size * self.config.valid_prop)
        train_set, valid_set = torch.utils.data.random_split(train_ts, [size - valid_size, valid_size])

        test_set = TimeSeries(test, self.config.history_w,
                              pred_w=self.config.predict_w,
                              stride=self.config.stride, device=self.device)
        anomaly_vect = anomaly_vect[len(train) + self.config.history_w:]
        assert len(test_set) == len(anomaly_vect), f"{len(test_set)} != {len(anomaly_vect)}"

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config.batch_size,
                                                   drop_last=True, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=self.config.batch_size, drop_last=True)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
        anomaly_vect_loader = torch.utils.data.DataLoader(anomaly_vect, batch_size=1)
        return PreparedData(train=train_loader, valid=valid_loader, test=(test_loader, anomaly_vect_loader))

    def __iter__(self):
        self.dataset_g = iter(self.dataset)
        return self

    def __next__(self):
        return self.run_once()

    def stats(self, name, loss_train, loss_valid, y_, y, label, to_html=True, html_file=None):
        (indexes,) = np.where(label == 1)
        intervals = utils.get_intevals(indexes)
        fig = make_subplots(rows=4, cols=2,
                            specs=[[{"colspan": 2}, None],
                                   [{"colspan": 2}, None],
                                   [{"rowspan": 2}, {"rowspan": 2}],
                                   [None, None]],
                            subplot_titles=["loss curve", "fit curve", "ROC curve", "Precision-Recall curve"])

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
        for x0, x1 in intervals:
            if x0 == x1:
                fig.add_shape(go.layout.Shape(type="line", xref='x', x0=x0, y0=min_y, x1=x0, y1=max_y,
                                              line={"dash": "dash", "color": "LightSkyBlue"}), row=2, col=1)
            else:
                fig.add_shape(go.layout.Shape(type="rect", xref='x', x0=x0, y0=min_y, x1=x1, y1=max_y,
                                              line={"color": "LightSkyBlue"}, fillcolor="LightSkyBlue", opacity=0.5),
                              row=2, col=1)

        prob = np.abs(y_ - y)

        fpr, tpr, _ = metrics.roc_curve(y_true=label, y_score=prob)
        roc_auc = metrics.auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", fill="tozeroy", line={"color": "#FF8E04"},
                                 showlegend=False), row=3, col=1)
        fig.add_shape(go.layout.Shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "blue"}),
                      row=3, col=1)
        fig.add_annotation(
            x=0.8,
            y=0.2,
            xref="x",
            yref="y",
            text=f"auc={roc_auc:.2f}",
            showarrow=False,
            align="center",
            bgcolor="#BDBDBD",
            opacity=0.8, row=3, col=1)

        precision, recall, _ = metrics.precision_recall_curve(y_true=label, probas_pred=prob)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", fill="tozeroy", line={"color": "#FF8E04"},
                                 showlegend=False), row=3, col=2)
        fig.add_shape(go.layout.Shape(type="line", x0=0, y0=1, x1=1, y1=0, line={"dash": "dash", "color": "blue"}),
                      row=3, col=2)

        fig.update_xaxes(title_text="False Positive Rate", row=3, col=1, range=[0, 1])
        fig.update_yaxes(title_text="True Positive Rate", row=3, col=1, range=[0, 1])
        fig.update_xaxes(title_text="Recall", row=3, col=2, range=[0, 1])
        fig.update_yaxes(title_text="Precision", row=3, col=2, range=[0, 1])

        label_pred = prob > self.config.sigma if not self.config.relative else (prob / y) > self.config.sigma
        label_pred.astype(int)
        tn, fp, fn, tp = metrics.confusion_matrix(label, label_pred).ravel()
        self.logger.info(f"tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}")
        prec, reca, f_beta, _ = metrics.precision_recall_fscore_support(label, label_pred, beta=self.config.beta,
                                                                        average="binary", zero_division=0)
        title = f"precision: {prec}, recall: {reca}, F beta score(beta={self.config.beta}): {f_beta}"
        self.logger.info(title)

        fig.update_layout(title=title)

        if to_html:
            fig.write_html(html_file if html_file is not None else f"./{self.config.res}/{name}_{self.timestamp}.html")
        else:
            fig.show()

        return prec, reca, f_beta
