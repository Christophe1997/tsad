from collections import OrderedDict

import torch
import math
import pyro.distributions as dist
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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, batch_first=True):
        super().__init__()
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe.requires_grad = False
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            return self.pe[:, :x.size(1)]
        else:
            return self.pe[:, :x.size(0)].transpose(0, 1)


class LGSSMPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=500, batch_first=True):
        super(LGSSMPositionalEncoding, self).__init__()
        self.batch_first = batch_first
        base_dist = dist.MultivariateNormal(torch.zeros(d_model), torch.eye(d_model))
        matrix = torch.eye(d_model)
        ghmm = dist.GaussianHMM(base_dist, matrix, base_dist, matrix, base_dist, duration=max_len)
        pe = ghmm.sample().unsqueeze(0)
        pe.requires_grad = False
        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.batch_first:
            # [b, l, n]
            return self.pe[:, :x.size(1)]
        else:
            # [l, b, n]
            return self.pe[:, :x.size(0)].transpose(0, 1)
