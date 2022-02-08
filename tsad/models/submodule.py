import copy
from collections import OrderedDict

import torch
import math
import pyro.distributions as dist
from torch import nn


class NormalParam(nn.Module):

    def __init__(self, input_dim, output_dim, eps=1e-5):
        super(NormalParam, self).__init__()
        self.x2mu = nn.Linear(input_dim, output_dim)
        self.x2scale = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Softplus()
        )
        self.eps = eps

    def forward(self, x, return_dist=False):
        loc = self.x2mu(x)
        scale = self.x2scale(x) + self.eps
        return (loc, scale) if not return_dist else (loc, scale, dist.Normal(loc, scale).to_event(1))


class MLPEmbedding(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_layers: list = None, dropout=0.1, activate=nn.Tanh()):
        super(MLPEmbedding, self).__init__()
        layers = OrderedDict()
        if hidden_layers is None:
            hidden_layers = []
        layer_dims = [input_dim] + hidden_layers + [output_dim]
        for i in range(1, len(layer_dims)):
            layers[f"linear_{i}"] = nn.Linear(layer_dims[i - 1], layer_dims[i], bias=False)
            layers[f"norm_{i}"] = nn.LayerNorm(layer_dims[i])
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


class MultiHeadLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True, layer_norm_eps=1e-5):
        super(MultiHeadLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mem, mask=None):
        res = self.norm(x)
        res = self.multihead_attn(res, mem, mem, attn_mask=mask, need_weights=False)[0]
        res = self.dropout(res)
        return x + res


class FeedforwardLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=1024, dropout=0.1, activate=nn.LeakyReLU()):
        super(FeedforwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activate = activate
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, batch_first=True, layer_norm_eps=1e-5):
        super(DecoderLayer, self).__init__()
        self.multihead = MultiHeadLayer(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.feedforward = FeedforwardLayer(d_model, dim_feedforward, dropout=dropout)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x, mem, mask=None):
        res = self.multihead(x, mem, mask=mask)
        res = self.feedforward(self.norm(res))
        return res


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, x, mem, mask=None):
        res = x
        for mod in self.layers:
            res = mod(res, mem, mask=mask)
        return res
