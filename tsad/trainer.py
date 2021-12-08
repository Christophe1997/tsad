import logging
import time

import torch.utils.data

from tsad.data import PreparedData
from tsad.wrapper import ModelWrapper


class Trainer:

    def __init__(self, model_wrapper: ModelWrapper, prepared_data: PreparedData, device=torch.device("cpu")):
        self.model_wrapper = model_wrapper
        self.prepared_data = prepared_data
        self.device = device
        self.logger = logging.getLogger("root")

        self.train_loader, self.valid_loader, self.test_loader = None, None, None

    def train(self, history_w, predict_w,
              epochs=100,
              batch_size=32,
              overlap=False,
              shuffle=True,
              test_batch_size=None, save_path=None):

        self.train_loader, self.valid_loader, self.test_loader = self.prepared_data.batchify(
            history_w, predict_w, batch_size, overlap, shuffle, test_batch_size, self.device)

        train_losses = []
        valid_losses = []
        best_valid_loss = None
        for epoch in range(epochs):
            start_time = time.time()
            train_loss = self.model_wrapper.train(self.train_loader)
            valid_loss = self.model_wrapper.eval(self.valid_loader)
            self.logger.info("epoch {:03d}, time {:0<6.2f}s, train_loss {:0<6.3f}, valid loss {:0<6.3f}".format(
                epoch + 1, time.time() - start_time, train_loss, valid_loss))
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if best_valid_loss is None or valid_loss < best_valid_loss:
                if save_path is not None:
                    self.model_wrapper.dump(save_path)
                best_valid_loss = valid_loss

        self.logger.info(f"best valid loss: {best_valid_loss:0<6.3f}")
        return train_losses, valid_losses

    def test(self, save_path=None):
        if save_path is not None:
            self.model_wrapper.load(save_path)

        test_loss = self.model_wrapper.eval(self.test_loader)
        self.logger.info(f"test loss: {test_loss:0<6.3f}")
        return test_loss
