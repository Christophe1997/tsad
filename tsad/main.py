import argparse
import logging
import os
import traceback

import numpy as np
import torch

from tsad.train import Train, EncoderTrain

parser = argparse.ArgumentParser("Train Script")
parser.add_argument("--data", type=str, default="./data/UCR_TimeSeriesAnomalyDatasets2021/UCR_Anomaly_FullData",
                    help="root dir of data")
parser.add_argument("--dataset", type=str, default="UCR",
                    help="dataset type(UCR, KPI, Yahoo), default 'UCR'")
parser.add_argument("--verbose", dest="log_level", action="store_const", const=logging.DEBUG, default=logging.INFO,
                    help="debug logging")
parser.add_argument("-O", "--output", type=str, default="model.pt", help="model save path, default 'model.pt'")
parser.add_argument("--seed", type=int, default=1234, help="random seed, default 1234")
parser.add_argument("--rnn_type", type=str, default="LSTM",
                    help="RNN type used for train(LSTM, GRU), default 'LSTM'")
parser.add_argument("--hidden_dim", type=int, default=512,
                    help="number of hidden units per layer, default 512")
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
parser.add_argument("--loss", type=str, default="MAE", help="loss function, default 'MAE'")
parser.add_argument("--optim", type=str, default="adam", help="optimizer, default 'adam'")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate, default 1e-3")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay, default 1e-4")
parser.add_argument("--clip", type=float, default=None, help="gradient clipping, default None")
parser.add_argument("--per_batch", type=int, default=10, help="log frequence per batch, default 10")
parser.add_argument("--res", type=str, default="out", help="log files and result files dir")
parser.add_argument("--one_file", type=str, default=None, help="only train on the specific data")
parser.add_argument("--sigma", type=float, default=None, help="sigma for anomaly detecting threshold")
parser.add_argument("--device", type=str, default="cuda", help="GPU device, if there is no gpu then use cpu")
parser.add_argument("--relative", default=False, action="store_true", help="use a relative error")
parser.add_argument("--beta", default=0.1, type=float, help="use for F beta score")
parser.add_argument("--emb_dim", default=64, type=int, help="embedding dimension for autoencoder")
parser.add_argument("--model_type", type=str, default="autoencoder", help="model type")

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

os.makedirs(args.res, exist_ok=True)
if args.model_type == "autoencoder":
    main = EncoderTrain(args)
else:
    main = Train(args)

# noinspection PyBroadException
try:
    if main.config.one_file is not None:
        res = main.run_once(main.config.one_file)
    else:
        for res in main:
            pass
except Exception:
    main.logger.error(traceback.format_exc())
