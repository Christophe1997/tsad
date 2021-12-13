import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from tsad.data import PickleDataset, PreparedData
from tsad.models.rnn import RNNAutoEncoder
from tsad.models.transformer import LinearDTransformer
from tsad.wrapper import LightningWrapper

import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")

parser = argparse.ArgumentParser("Train Script")
parser.add_argument("--data", type=str, default="./data/KPI",
                    help="root dir of data")
parser.add_argument("--dataset", type=str, default="KPI",
                    help="dataset type(KPI, MSL, SMAP, SMD), default 'KPI'")
parser.add_argument("-O", "--output", type=str, default="out", help="result dir")
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
parser.add_argument("--one_file", type=str, default=None, help="only train on the specific data")
parser.add_argument("--gpu", type=int, default=-1, help="GPU device, if there is no gpu then use cpu")
parser.add_argument("--emb_dim", default=64, type=int, help="embedding dimension for autoencoder")
parser.add_argument("--model_type", type=str, default="autoencoder", help="model type")
parser.add_argument("--dim_feedforward", type=int, default=1024, help="dimension of transformer feedforward layer")
parser.add_argument("--no_progress_bar", dest="enable_progress_bar", action="store_false",
                    help="whether enable progress_bar")


def get_dataset(dataset_type, dir_path):
    if dataset_type in ["KPI", "SMD"]:
        return PickleDataset(dir_path, prefix=dataset_type)
    elif dataset_type in ["MSL", "SMAP"]:
        return PickleDataset(dir_path, indices=[dataset_type], prefix=dataset_type)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def train(prepared_data, args):
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    logger = TensorBoardLogger(args.output, name=f"{prepared_data.data_id}_{args.model_type}", default_hp_metric=False)
    early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=1e-4, patience=5)
    if args.gpu >= 0:
        gpus = [args.gpu]
    else:
        gpus = 0
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[early_stop_callback],
        gpus=gpus,
        enable_progress_bar=args.enable_progress_bar)

    train_loader, valid_loader, test_loader = prepared_data.batchify(
        history_w=args.history_w,
        predict_w=0,
        batch_size=args.batch_size,
        overlap=True,
        shuffle=True,
        test_batch_size=None,
        device=device)
    if args.model_type == "autoencoder":
        wrapper = LightningWrapper(
            RNNAutoEncoder,
            window_size=args.history_w,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            rnn_type=args.rnn_type,
            n_features=prepared_data.n_features)
        trainer.gradient_clip_val = 2
        trainer.fit(wrapper, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        trainer.test(dataloaders=test_loader, ckpt_path="best")

    elif args.model_type == "linearTransformer":
        root = f"{args.output}/{prepared_data.data_id}_autoencoder/"
        last_version = sorted(os.listdir(root))[-1]
        root = os.path.join(root, last_version, "checkpoints")
        ckpt_path = os.path.join(root, os.listdir(root)[0])
        print(ckpt_path)
        encoder = LightningWrapper.load_from_checkpoint(ckpt_path, map_location=device).model.encoder
        wrapper = LightningWrapper(
            LinearDTransformer,
            d_model=args.emb_dim,
            history_w=args.history_w,
            predict_w=args.predict_w,
            encoder=encoder,
            dim_feedforward=256,
            n_features=prepared_data.n_features,
            dropout=args.dropout,
            overlap=True)
        trainer.fit(wrapper, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        trainer.test(dataloaders=test_loader, ckpt_path="best")
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


# noinspection PyBroadException
def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = get_dataset(args.dataset, args.data)

    if args.one_file is not None:
        prepared_data = PreparedData(*dataset.load_one(args.one_file), valid_prop=args.valid_prop)
        train(prepared_data, args)
    else:
        for item in dataset:
            prepared_data = PreparedData(*item, valid_prop=args.valid_prop)
            train(prepared_data, args)


if __name__ == "__main__":
    args_ = parser.parse_args()
    main(args_)
