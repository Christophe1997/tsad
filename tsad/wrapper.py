import pytorch_lightning as pl
import torch
from torch import optim
from torch.nn import functional


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
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
        return optimizer

    def criterion(self, y_actul, y_pred):
        if self.loss_type == "MAE":
            return functional.l1_loss(y_actul, y_pred, reduction="mean")
