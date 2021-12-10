import logging

import torch
from torch import nn, optim


class ModelWrapper:

    def __init__(self, model: nn.Module, device=torch.device("cpu"), lr=1e-3, weight_decay=0.01):
        self.device = device
        self.model = model.to(device)
        self.model_type = model.__class__.__name__
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()
        self.logger = logging.getLogger("root")

    @classmethod
    def from_static(cls, path, device=torch.device("cpu"), lr=1e-3, weight_decay=0.01):
        with open(path, "rb") as f:
            model = torch.load(f)

        return cls(model, device, lr, weight_decay)

    def get_optimizer(self):
        return optim.AdamW(self.model.parameters(), self.lr, weight_decay=self.weight_decay)

    def get_criterion(self):
        return nn.L1Loss(reduction="mean").to(self.device)

    def train(self, dataloader):
        self.model.train()
        total_loss = 0
        total_batches = len(dataloader)

        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(x_batch), y_batch)
            loss.backward()
            self.train_extra()
            self.optimizer.step()
            total_loss += loss.item()
            self.logger.debug("{:05d}/{:05d} batches, loss {:0<6.3f}".format(
                batch_idx + 1, total_batches, loss.item()))

        return total_loss / total_batches

    def train_extra(self):
        pass

    def eval(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_batches = len(dataloader)
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                loss = self.criterion(self.model(x_batch), y_batch)
                total_loss += loss.item()
                self.logger.debug("{:05d}/{:05d} batches, loss {:0<6.3f}".format(
                    batch_idx + 1, total_batches, loss.item()))

        return total_loss / total_batches

    def predict(self, dataloader):
        self.model.eval()
        actual, pred = [], []
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                actual.append(y_batch)
                pred.append(self.model(x_batch))

        actual = torch.vstack(actual)
        pred = torch.vstack(pred)

        return actual, pred

    def dump(self, file):
        with open(file, "wb") as f:
            torch.save(self.model, f)

    def load(self, file):
        with open(file, "rb") as f:
            self.model = torch.load(f)


class RNNModelWrapper(ModelWrapper):

    def __init__(self, model, device, clip=10):
        super(RNNModelWrapper, self).__init__(model, device)
        self.clip = clip

    def train_extra(self):
        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
