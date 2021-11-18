import argparse
import logging
from tsad import utils
from tsad.model import RNNModel

parser = argparse.ArgumentParser("Train Script")
parser.add_argument("--data", type=str, help="root dir of data")
parser.add_argument("--model", type=str, default="LSTM", help="model type")
parser.add_argument("--verbose", dest="log_level", action="store_const", const=logging.DEBUG,
                    help="debug logging")
parser.set_defaults(log_level=logging.INFO)
parser.add_argument("-O", "--output", type=str, default="model.pt", help="model save path")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--rnn_type", type=str, default="LSTM",
                    help="RNN type used for train")
parser.add_argument("--hidden_dim", type=int, default=256,
                    help="number of hidden units per layer")
parser.add_argument("--num_layer", type=int, default=1,
                    help="number of hidden layers")
parser.add_argument("--history_w", type=int, default=45,
                    help="history window size for predicting")
parser.add_argument("predict_w", type=int, default=1,
                    help="predict window size")
parser.add_argument("dropout", type=int, default=0.5,
                    help="dropout applied to layers")
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--epochs", type=int, default=50, help="epoch limit")

args = parser.parse_args()
