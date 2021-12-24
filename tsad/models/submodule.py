from collections import OrderedDict

import torch
import math
import pyro.distributions as dist
from torch import nn
from torch.nn import functional


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
        layers = OrderedDict()
        if hidden_layers is None:
            hidden_layers = []
        layer_dims = [input_dim] + hidden_layers + [output_dim]
        for i in range(1, len(layer_dims)):
            layers[f"linear_{i}"] = nn.Linear(layer_dims[i - 1], layer_dims[i])
            layers[f"activate_{i}"] = activate
            layers[f"dropout_{i}"] = nn.Dropout(dropout)

        self.embeding = nn.Sequential(layers)

    def forward(self, x):
        return self.embeding(x)


class Conv1DEmbedding(nn.Module):

    def __init__(self, input_dim, output_dim, padding="causal", kernel_size=3, batch_first=True):
        super(Conv1DEmbedding, self).__init__()
        self.batch_first = batch_first

        pad_size = (kernel_size - 1)
        if padding == "causal":
            layers = OrderedDict({
                "pad_0": nn.ConstantPad1d((pad_size, 0), 0),
                "conv_0": nn.Conv1d(input_dim, output_dim, (kernel_size,))
            })
            self.conv = nn.Sequential(layers)

        elif padding == "noncausal":
            self.conv = nn.Conv1d(input_dim, output_dim, (kernel_size,), padding=pad_size // 2)
        else:
            raise ValueError(f"wrong padding param: {padding}")

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.permute(0, 2, 1) if self.batch_first else x.permute(1, 2, 0)
        res = self.conv(x)
        return res.permute(0, 2, 1) if self.batch_first else res.permute(2, 0, 1)


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


class TimePositionalEncoding(nn.Module):
    """Use LGSSM(linear gaussian state space model) to produce timestamp position encoding.

    Which works as follow:
        z = initial_dist.sample()
        x = []
        for t in range(timestep):
            z = z @ transition_matrix + transition_dist.sample()
            x.append(z @ observation_matrix + observation_dist.sample())
    So, in this way, step t encoding is produced by {1, ..., t-1} step state
    """

    def __init__(self, d_model, max_len=500, batch_first=True):
        super(TimePositionalEncoding, self).__init__()
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
