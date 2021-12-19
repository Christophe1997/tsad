import pytorch_lightning as pl
import torch
from torch import optim
from torch.nn import functional
import pyro
import pyro.optim


# noinspection PyAbstractClass
class LightningWrapper(pl.LightningModule):

    def __init__(self, model_cls, loss_type="MAE", *args, **kwargs):
        super(LightningWrapper, self).__init__()
        self.model = model_cls(*args, **kwargs)
        self.model_type = self.model.__class__.__name__
        self.loss_type = loss_type
        self.save_hyperparameters()

    def forward(self, dataloader):
        actual, pred = [], []
        for x_batch, y_batch in dataloader:
            actual.append(y_batch)
            pred.append(self.model(x_batch))

        actual = torch.vstack(actual)
        pred = torch.vstack(pred)

        return actual, pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.model(x)
        loss = self.criterion(y_, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.model(x)
        loss = self.criterion(y_, y)
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.model(x)
        loss = self.criterion(y_, y)
        return loss

    def test_epoch_end(self, outputs) -> None:
        loss = torch.stack(outputs)
        self.log("test_loss", torch.mean(loss))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
        return optimizer

    def criterion(self, y_actul, y_pred):
        if self.loss_type == "MAE":
            return functional.l1_loss(y_actul, y_pred, reduction="mean")


# noinspection PyAbstractClass
class PyroLightningWrapper(pl.LightningModule):
    class PyrOptim(pyro.infer.SVI, torch.optim.Optimizer):
        def __init__(self, *args, **kwargs):
            super(PyroLightningWrapper.PyrOptim, self).__init__(*args, **kwargs)
            self.state = {}

        def state_dict(self):
            return {}

        def __setstate__(self, state):
            super(PyroLightningWrapper.PyrOptim, self).__setstate__(state)

    def __init__(self, model_cls, *args, **kwargs):
        super(PyroLightningWrapper, self).__init__()
        self.model = model_cls(*args, **kwargs)
        self.model_type = self.model.__class__.__name__
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.min_annealing_factor = 0.2
        self.annealing_epochs = 100
        self.epoch_idx = 0
        self.num_batches = None

    def get_anneling_factor(self, batch_idx):
        if self.epoch_idx < self.annealing_epochs and self.num_batches is not None:
            return self.min_annealing_factor + (1 - self.min_annealing_factor) * (
                    (batch_idx + 1 + self.epoch_idx * self.num_batches) / (self.annealing_epochs * self.num_batches))
        else:
            return 1.0

    def forward(self, dataloader):
        actual, pred = [], []
        for x_batch, y_batch in dataloader:
            actual.append(y_batch)
            pred.append(self.model(x_batch))

        actual = torch.vstack(actual)
        pred = torch.vstack(pred)

        return actual, pred

    def training_step(self, batch, batch_idx):
        x, _ = batch
        optimizer = self.optimizers(use_pl_optimizer=False)
        loss = optimizer.step(x, annealing_factor=self.get_anneling_factor(batch_idx))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def training_epoch_end(self, outputs) -> None:
        self.epoch_idx += 1

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        optimizer = self.optimizers(use_pl_optimizer=False)
        loss = optimizer.evaluate_loss(x)
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        optimizer = self.optimizers(use_pl_optimizer=False)
        loss = optimizer.evaluate_loss(x)
        return loss

    def test_epoch_end(self, outputs):
        loss = torch.stack(outputs)
        self.log("test loss", torch.mean(loss))

    def configure_optimizers(self):
        loss = pyro.infer.Trace_ELBO()
        svi = self.PyrOptim(
            model=self.model.model,
            guide=self.model.guide,
            optim=pyro.optim.ClippedAdam({
                "lr": 3e-4,
                "betas": (0.95, 0.999),
                "clip_norm": 10,
                "lrd": 0.99996,
                "weight_decay": 2}),
            loss=loss)

        return svi


# noinspection PyAbstractClass
class DvaeLightningWrapper(pl.LightningModule):
    def __init__(self, model_cls, *args, **kwargs):
        super(DvaeLightningWrapper, self).__init__()
        self.model = model_cls(*args, **kwargs)
        self.model_type = self.model.__class__.__name__
        self.save_hyperparameters()

        self.min_annealing_factor = 0.2
        self.annealing_epochs = 50
        self.epoch_idx = 0
        self.num_batches = None

    def get_anneling_factor(self, batch_idx):
        if self.epoch_idx < self.annealing_epochs and self.num_batches is not None:
            return self.min_annealing_factor + (1 - self.min_annealing_factor) * (
                    (batch_idx + 1 + self.epoch_idx * self.num_batches) / (self.annealing_epochs * self.num_batches))
        else:
            return 1.0

    def forward(self, dataloader):
        actual, pred = [], []
        for x_batch, y_batch in dataloader:
            actual.append(y_batch)
            pred.append(self.model(x_batch))

        actual = torch.vstack(actual)
        pred = torch.vstack(pred)

        return actual, pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.model(x)
        anneling_factor = self.get_anneling_factor(batch_idx)
        loss = self.criterion(y, y_, anneling_factor=anneling_factor)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.model(x)
        loss = self.criterion(y, y_)
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.model(x)
        loss = self.criterion(y, y_)
        return loss

    def test_epoch_end(self, outputs) -> None:
        loss = torch.stack(outputs)
        self.log("test_loss", torch.mean(loss))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=3e-4, weight_decay=2, betas=(0.95, 0.999))
        return optimizer

    def criterion(self, y_actul, y_pred, anneling_factor=1.0):
        return self.model.compute_loss(y_actul, y_pred, anneling_factor=anneling_factor)
