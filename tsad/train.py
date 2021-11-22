import argparse
import logging
from tsad import utils
from tsad.model import RNNModel
from tsad.data import UCRTSAD2021Dataset
from torch import nn
from torch import optim

parser = argparse.ArgumentParser("Train Script")
parser.add_argument("--data", type=str, default="./data/UCR_TimeSeriesAnomalyDatasets2021/UCR_Anomaly_FullData",
                    help="root dir of data")
parser.add_argument("--dataset", type=str, default="UCR", help="dataset type")
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
parser.add_argument("--dropout", type=int, default=0.5,
                    help="dropout applied to layers")
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--epochs", type=int, default=50, help="epoch limit")
parser.add_argument("--loss", type=str, default="L1", help="loss function")
parser.add_argument("--optim", type=str, default="adam", help="optimizer")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")

args = parser.parse_args()


class Train:

    def __init__(self, config):
        self.config = config
        self.model = self.get_model()
        self.dataset = self.get_data()
        self.criterion = self.get_loss()
        self.optimizer = self.get_optim(self.model)

    def get_model(self):
        if self.config.rnn_type in ["LSTM", "GRU"]:
            return RNNModel(self.config.history_w, self.config.predict_w, self.config.hidden_dim,
                            self.config.num_layers, self.config.dropout, self.config.rnn_type)

    def get_loss(self):
        if self.config.loss == "L1":
            return nn.L1Loss()

    def get_optim(self, model: nn.Module):
        if self.config.optim == "adam":
            return optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

    def get_data(self):
        if self.config.dataset == "UCR":
            return UCRTSAD2021Dataset(self.config.data)

    def run_once(self, epochs):
        pass
