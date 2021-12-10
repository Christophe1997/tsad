import argparse
import datetime
import logging
import os
import time
import traceback

import numpy as np
import torch

from tsad.data import UCRTSAD2021Dataset, KPIDataset, YahooS5Dataset, PreparedData
from tsad import utils
from tsad.wrapper import RNNModelWrapper, ModelWrapper
from tsad.models.rnn import RNNAutoEncoder
from tsad.models.transformer import LinearDTransformer
from tsad.trainer import Trainer

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
parser.add_argument("--hidden_dim", type=int, default=128,
                    help="number of hidden units per layer, default 128")
parser.add_argument("--num_layers", type=int, default=2,
                    help="number of hidden layers, default 2")
parser.add_argument("--history_w", type=int, default=32,
                    help="history window size for predicting, default 32")
parser.add_argument("--predict_w", type=int, default=1,
                    help="predict window size, default 1")
parser.add_argument("--dropout", type=float, default=0.2,
                    help="dropout applied to layers, default 0.2")
parser.add_argument("--batch_size", type=int, default=32, help="batch size, default 32")
parser.add_argument("--epochs", type=int, default=100, help="epoch limit, default 100")
parser.add_argument("--valid_prop", type=float, default=0.3, help="validation set prop, default 0.3")
parser.add_argument("--test_prop", type=float, default=0.5,
                    help="test set prop(only work for some dataset), default 0.5")
parser.add_argument("--res", type=str, default="out", help="log files and result files dir")
parser.add_argument("--one_file", type=str, default=None, help="only train on the specific data")
parser.add_argument("--device", type=str, default="cuda", help="GPU device, if there is no gpu then use cpu")
parser.add_argument("--emb_dim", default=64, type=int, help="embedding dimension for autoencoder")
parser.add_argument("--model_type", type=str, default="autoencoder", help="model type")
parser.add_argument("--dim_feedforward", type=int, default=1024, help="dimension of transformer feedforward layer")


def get_dataset(dataset_type, dir_path):
    if dataset_type == "UCR":
        return UCRTSAD2021Dataset(dir_path)
    elif dataset_type == "KPI":
        return KPIDataset(dir_path)
    elif dataset_type == "Yahoo":
        return YahooS5Dataset(dir_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def train(prepared_data, args, device):
    logger = logging.getLogger("root")
    start = time.time()
    if args.model_type == "autoencoder":
        model = RNNAutoEncoder(window_size=args.history_w,
                               emb_dim=args.emb_dim,
                               hidden_dim=args.hidden_dim,
                               rnn_type=args.rnn_type,
                               n_features=1)
        wrapper = RNNModelWrapper(model, device)
        trainer = Trainer(wrapper, prepared_data, device)
        path = f"{args.res}/{prepared_data.data_id}_{args.output}"
        train_losses, valid_losses = trainer.train(
            history_w=args.history_w,
            predict_w=0,
            epochs=args.epochs,
            batch_size=args.batch_size,
            overlap=True,
            shuffle=True,
            test_batch_size=None,
            save_path=path)
        test_loss = trainer.test(path)

    elif args.model_type == "linearTransformer":
        path = f"{args.res}/{prepared_data.data_id}_{args.output}"
        with open(path, "rb") as f:
            encoder = torch.load(f, map_location=device).encoder
        model = LinearDTransformer(d_model=args.emb_dim,
                                   history_w=args.history_w,
                                   predict_w=args.predict_w,
                                   encoder=encoder,
                                   dim_feedforward=256,
                                   dropout=args.dropout,
                                   overlap=True)
        wrapper = ModelWrapper(model, device)
        trainer = Trainer(wrapper, prepared_data, device)
        path = f"{args.res}/{prepared_data.data_id}_ltf_{args.output}"
        train_losses, valid_losses = trainer.train(
            history_w=args.history_w,
            predict_w=args.predict_w,
            epochs=args.epochs,
            batch_size=args.batch_size,
            overlap=True,
            shuffle=True,
            test_batch_size=None,
            save_path=path)
        test_loss = trainer.test(path)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    logger.info(f"{prepared_data.data_id}: total cost {time.time() - start:.0f}s, "
                f"train loss {train_losses[0]:0<6.4f} -> {train_losses[-1]:0<6.4f}, "
                f"valid loss {valid_losses[0]:0<6.4f} -> {valid_losses[-1]:0<6.4f}, "
                f"test loss {test_loss:0<6.4f}")


# noinspection PyBroadException
def main(main_id):
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(args.res, exist_ok=True)
    logger = utils.get_logger(args, f"{args.res}/{main_id}.log")

    logger.info(utils.dict2str(vars(args)))
    logger.info(f"main ID: {main_id}")
    dataset = get_dataset(args.dataset, args.data)
    try:
        if args.one_file is not None:
            prepared_data = PreparedData(*dataset.load_one(args.one_file), valid_prop=args.valid_prop)
            train(prepared_data, args, device)
        else:
            for item in dataset:
                prepared_data = PreparedData(*item, valid_prop=args.valid_prop)
                train(prepared_data, args, device)
    except Exception:
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    timestamp = int(datetime.datetime.now().timestamp())
    main(timestamp)
