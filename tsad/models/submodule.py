from collections import OrderedDict

import torch
from torch import nn


class NormalParam(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(NormalParam, self).__init__()
        self.mu = nn.Linear(input_dim, output_dim)
        self.logvar = nn.Linear(input_dim, output_dim)

    def forward(self, x, return_logvar=False):
        loc = self.mu(x)
        logvar = self.logvar(x)
        if return_logvar:
            return loc, logvar
        else:
            scale = torch.exp(0.5 * logvar)
            return loc, scale


class MLPEmbedding(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_layers: list = None, dropout=0.1, activate=nn.Tanh()):
        super(MLPEmbedding, self).__init__()
        dict_layers = OrderedDict()
        if hidden_layers is None:
            hidden_layers = []
        layer_dims = [input_dim] + hidden_layers + [output_dim]
        for i in range(1, len(layer_dims)):
            dict_layers[f"linear_{i}"] = nn.Linear(layer_dims[i - 1], layer_dims[i])
            dict_layers[f"activate_{i}"] = activate
            dict_layers["dropout_{i}"] = nn.Dropout(dropout)

        self.embeding = nn.Sequential(dict_layers)

    def forward(self, x):
        return self.embeding(x)
