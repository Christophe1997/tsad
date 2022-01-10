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
from tsad.models.vae import VRNNPyro, RVAEPyro, SRNNPyro, OmniPyro, TransformerVAEPyro
from tsad.models.dvae import VRNN, RVAE, SRNN, Omni, TransformerVAE
from tsad.wrapper import DvaeLightningWrapper, PyroLightningWrapper

models = [
    VRNN, VRNNPyro,
    RVAE, RVAEPyro,
    SRNN, SRNNPyro,
    Omni, OmniPyro,
    TransformerVAE, TransformerVAEPyro
]
config = Config()
logger = logging.getLogger("pytorch_lightning")
logger.setLevel(logging.INFO)
os.makedirs("./log", exist_ok=True)
logger.addHandler(logging.FileHandler(f"./log/{datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}.log"))
warnings.filterwarnings("ignore", ".*does not have many workers.*")


def get_dataset(dataset_type, dir_path):
    if dataset_type in ["MSL", "SMAP"]:
        return PickleDataset(dir_path, indices=[dataset_type], prefix=dataset_type)
    else:
        return PickleDataset(dir_path, prefix=dataset_type)


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
        min_epochs=args.min_epochs,
        max_epochs=args.epochs,
        logger=tb_logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        gpus=gpus,
        gradient_clip_val=args.clip,
        enable_progress_bar=args.enable_progress_bar)

    train_loader, valid_loader, test_loader = prepared_data.batchify(
        history_w=args.history_w,
        predict_w=0,
        batch_size=args.batch_size,
        overlap=True,
        shuffle=True,
        test_batch_size=None,
        num_workers=args.num_workers)

    ckpt_rootdir = f"{args.output}/{prepared_data.data_id}_{args.model_type}/"
    model_type = None
    if args.with_pyro:
        wrapper_cls = PyroLightningWrapper
        model_type = f"{args.model_type}_pyro"
        trainer.gradient_clip_val = None
        pyro.clear_param_store()
    else:
        wrapper_cls = DvaeLightningWrapper

    def train_aux(**kwargs):
        best_model_path = None
        if not args.test_only:
            model_cls, model_params = config.load_config(args, model_type=model_type, **kwargs)
            wrapper = wrapper_cls(model_cls, **model_params)
            logger.info(f"Pyro: {isinstance(wrapper, PyroLightningWrapper)}")
            logger.info(wrapper.model)
            wrapper.num_batches = len(train_loader)
            trainer.fit(wrapper, train_dataloaders=train_loader, val_dataloaders=valid_loader)
            best_model_path = checkpoint_callback.best_model_path

        if best_model_path is None:
            best_model_path = utils.get_last_ckpt(ckpt_rootdir)
        wrapper = wrapper_cls.load_from_checkpoint(best_model_path)
        wrapper.model = wrapper.model.to(device)
        y_score, y_loc, y_scale = utils.get_score(wrapper, test_loader, n_sample=10, device=device)
        return y_score, y_loc, y_scale

    scores, y_locs, y_scales = train_aux(n_features=prepared_data.n_features,
                                         nhead=8, nlayers=2  # default  for transformer net
                                         )
    anomaly_vect = prepared_data.test_anomaly[args.history_w - 1:]
    scores = scores.cpu().numpy()
    y_locs = utils.reconstruct(y_locs.cpu().numpy())
    y_scales = utils.reconstruct(y_scales.cpu().numpy())

    last_version = utils.get_last_version(ckpt_rootdir)
    with open(os.path.join(ckpt_rootdir, f"{last_version}/test_score.pkl"), "wb") as f:
        pickle.dump(scores, f)
    with open(os.path.join(ckpt_rootdir, f"{last_version}/test_mean.pkl"), "wb") as f:
        pickle.dump(y_locs, f)
    with open(os.path.join(ckpt_rootdir, f"{last_version}/test_std.pkl"), "wb") as f:
        pickle.dump(y_scales, f)

    fpr, tpr, thresholds = metrics.roc_curve(y_true=anomaly_vect, y_score=scores)
    roc_auc = metrics.auc(fpr, tpr)
    if not args.test_only:
        tb_logger.log_metrics({"hp_metric": roc_auc})

    non_anomaly_indices = np.where(prepared_data.test_anomaly != 1)
    rmse = metrics.mean_squared_error(prepared_data.test[non_anomaly_indices], y_locs[non_anomaly_indices],
                                      squared=False)
    logger.info(
        f"RMSE(with mean val) = {rmse:.3f} ROC auc = {roc_auc:.3f} on {prepared_data.data_id} with {args.model_type}")


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
    parser.add_argument("--dataset", type=str, default="SMD",
                        help="dataset type(ASD, MSL, SMAP, SMD), default 'SMD'")
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
                        help="model type('vrnn', 'omni', 'rvae', 'srnn', 'tfvae')")
    parser.add_argument("--dense_dim", type=int, default=512, help="dimension of transformer feedforward layer")
    parser.add_argument("--no_progress_bar", dest="enable_progress_bar", action="store_false",
                        help="whether enable progress_bar")
    parser.add_argument("--test_only", action="store_true", help="test only")
    parser.add_argument("--with_phi_dense", dest="phi_dense", action="store_true",
                        help="add dense layers in inference net")
    parser.add_argument("--with_theta_dense", dest="theta_dense", action="store_true",
                        help="add dense layers in generation net")
    parser.add_argument("--with_pyro", action="store_true", help="use pyro for train")
    parser.add_argument("--num_workers", type=int, default=0, help="dataload num_worker setting")
    parser.add_argument("--min_epochs", type=int, default=20, help="min epochs")
    parser.add_argument("--clip", type=float, default=10, help="gradient clip")

    args_ = parser.parse_args()
    # noinspection PyBroadException
    try:
        main(args_)
    except Exception:
        logger.error(traceback.format_exc())
