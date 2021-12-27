import argparse
import datetime
import logging
import os
import pickle
import traceback
import warnings

import numpy as np
import pyro
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn import metrics

from tsad import utils
from tsad.data import PickleDataset, PreparedData
from tsad.config import Config
from tsad.models.vae import VRNNPyro
from tsad.models.dvae import VRNN
from tsad.wrapper import DvaeLightningWrapper, PyroLightningWrapper

models = [VRNN, VRNNPyro]
config = Config()
logger = logging.getLogger("pytorch_lightning")
logger.setLevel(logging.INFO)
os.makedirs("./log", exist_ok=True)
logger.addHandler(logging.FileHandler(f"./log/{datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}.log"))
warnings.filterwarnings("ignore", ".*does not have many workers.*")


def get_dataset(dataset_type, dir_path):
    if dataset_type in ["KPI", "SMD"]:
        return PickleDataset(dir_path, prefix=dataset_type)
    elif dataset_type in ["MSL", "SMAP"]:
        return PickleDataset(dir_path, indices=[dataset_type], prefix=dataset_type)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def train(prepared_data, args):
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    tb_logger = TensorBoardLogger(args.output, name=f"{prepared_data.data_id}_{args.model_type}")

    checkpoint_callback = ModelCheckpoint(monitor="valid_loss", filename='{epoch:02d}-{valid_loss:.2f}')
    early_stop_callback = EarlyStopping(monitor="valid_loss_epoch", min_delta=2, patience=3)
    if args.gpu >= 0:
        gpus = [args.gpu]
    else:
        gpus = 0
    trainer = pl.Trainer(
        min_epochs=20,
        max_epochs=args.epochs,
        logger=tb_logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        gpus=gpus,
        gradient_clip_val=10,
        deterministic=True,
        enable_progress_bar=args.enable_progress_bar)

    train_loader, valid_loader, test_loader = prepared_data.batchify(
        history_w=args.history_w,
        predict_w=args.predict_w,
        batch_size=args.batch_size,
        overlap=True,
        shuffle=True,
        test_batch_size=None,
        device=device)

    ckpt_rootdir = f"{args.output}/{prepared_data.data_id}_{args.model_type}/"

    if args.with_pyro or args.model_type.endswith("pyro"):
        wrapper_cls = PyroLightningWrapper
    else:
        wrapper_cls = DvaeLightningWrapper

    def train_aux(**kwargs):
        best_model_path = None
        if not args.test_only:
            trainer.gradient_clip_val = None
            model_cls, model_params = config.load_config(args, **kwargs)
            print(model_params)
            wrapper = wrapper_cls(model_cls, **model_params)
            wrapper.num_batches = len(train_loader)
            pyro.clear_param_store()
            trainer.fit(wrapper, train_dataloaders=train_loader, val_dataloaders=valid_loader)
            best_model_path = checkpoint_callback.best_model_path

        if best_model_path is None:
            best_model_path = utils.get_last_ckpt(ckpt_rootdir)
        wrapper = wrapper_cls.load_from_checkpoint(best_model_path)
        wrapper.model = wrapper.model.to(device)
        res = utils.get_score(wrapper, test_loader)
        return res

    scores = train_aux(n_features=prepared_data.n_features,
                       feature_x=64, feature_z=32,  # default for rnn net
                       nhead=8, nlayers=2, phi_dense=False, theta_dense=False  # default  for transformer net
                       )
    anomaly_vect = prepared_data.test_anomaly[args.history_w - 1:]
    scores = scores.cpu().numpy()

    last_version = sorted(os.listdir(ckpt_rootdir))[-1]
    with open(os.path.join(ckpt_rootdir, f"{last_version}_test_score.pkl"), "wb") as f:
        pickle.dump(scores, f)

    fpr, tpr, thresholds = metrics.roc_curve(y_true=anomaly_vect, y_score=scores)
    roc_auc = metrics.auc(fpr, tpr)
    if not args.test_only:
        tb_logger.log_metrics({"hp_metric": roc_auc})
    logger.info(f"ROC auc = {roc_auc:.3f} on {prepared_data.data_id} with {args.model_type}")


# noinspection PyBroadException
def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    pyro.set_rng_seed(args.seed)

    dataset = get_dataset(args.dataset, args.data)

    if args.one_file is not None:
        prepared_data = PreparedData(*dataset.load_one(args.one_file), valid_prop=args.valid_prop)
        train(prepared_data, args)
    else:
        for item in dataset:
            prepared_data = PreparedData(*item, valid_prop=args.valid_prop)
            train(prepared_data, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Script")
    parser.add_argument("--data", type=str, default="./data/KPI",
                        help="root dir of data")
    parser.add_argument("--dataset", type=str, default="KPI",
                        help="dataset type(KPI, MSL, SMAP, SMD), default 'KPI'")
    parser.add_argument("-O", "--output", type=str, default="out", help="result dir")
    parser.add_argument("--seed", type=int, default=1234, help="random seed, default 1234")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="number of hidden units per layer, default 128")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers, default 2")
    parser.add_argument("--history_w", type=int, default=32,
                        help="history window size for predicting, default 32")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout applied to layers, default 0.2")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size, default 32")
    parser.add_argument("--epochs", type=int, default=100, help="epoch limit, default 100")
    parser.add_argument("--valid_prop", type=float, default=0.3, help="validation set prop, default 0.3")
    parser.add_argument("--test_prop", type=float, default=0.5,
                        help="test set prop(only work for some dataset), default 0.5")
    parser.add_argument("--one_file", type=str, default=None, help="only train on the specific data")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU device, if there is no gpu then use cpu")
    parser.add_argument("--z_dim", default=64, type=int, help="embedding dimension for autoencoder")
    parser.add_argument("--model_type", type=str, default="vrnn",
                        help="model type('vrnn', 'vrnn_pyro')")
    parser.add_argument("--dim_dense", type=int, default=512, help="dimension of transformer feedforward layer")
    parser.add_argument("--no_progress_bar", dest="enable_progress_bar", action="store_false",
                        help="whether enable progress_bar")
    parser.add_argument("--test_only", action="store_true", help="test only")
    parser.add_argument("--with_pyro", action="store_true", help="use pyro for train")

    args_ = parser.parse_args()
    # noinspection PyBroadException
    try:
        main(args_)
    except Exception:
        logger.error(traceback.format_exc())
