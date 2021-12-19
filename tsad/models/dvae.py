import torch

from torch import nn
from collections import OrderedDict
from tsad.models.vae import NormalParam
from torch.nn import functional


def reparameterization(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return torch.addcmul(mean, eps, std)


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


class VRNN(nn.Module):
    """ Original paper: A Recurrent Latent Variable Model for Sequential Data (https://arxiv.org/abs/1506.02216)
    """

    def __init__(self, n_features=1, hidden_dim=256, z_dim=4, dropout=0.1):
        super(VRNN, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.dim_feature_x = hidden_dim // 2
        self.dim_feature_z = 16

        self.rnn = nn.GRUCell(input_size=self.dim_feature_x + z_dim, hidden_size=hidden_dim)
        self.phi_norm = NormalParam(hidden_dim + self.dim_feature_x, z_dim)
        self.theta_norm_1 = NormalParam(hidden_dim, z_dim)
        self.theta_norm_2 = NormalParam(self.dim_feature_z + hidden_dim, n_features)

        self.feature_extra_x = MLPEmbedding(n_features, self.dim_feature_x, dropout=dropout)
        self.feature_extra_z = MLPEmbedding(z_dim, self.dim_feature_z, [self.dim_feature_z * 2], dropout=dropout)

        # placeholder
        self.z_loc = None
        self.z_logvar = None
        self.z = None
        self.z_loc_prior = None
        self.z_logvar_prior = None

    def encode(self, x, h):
        h_with_x = torch.cat([h, x], dim=1)
        z_loc, z_logvar = self.phi_norm(h_with_x, return_logvar=True)
        z = reparameterization(z_loc, z_logvar)
        return z, z_loc, z_logvar

    def decode(self, z, h, return_prob=False):
        feature_z = self.feature_extra_z(z)
        z_with_h = torch.cat([feature_z, h], dim=1)
        x_loc, x_scale = self.theta_norm_2(z_with_h)
        return x_loc if not return_prob else (x_loc, x_scale)

    def recurrence(self, x, z, h):
        x_with_z = torch.cat([x, z], dim=1)
        h = self.rnn(x_with_z, h)
        return h

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        self.z_loc = x.new_zeros([batch_size, seq_len, self.z_dim])
        self.z_logvar = x.new_zeros([batch_size, seq_len, self.z_dim])
        y = x.new_zeros([batch_size, seq_len, self.n_features])
        self.z = x.new_zeros([batch_size, seq_len, self.z_dim])

        h = x.new_zeros([batch_size, seq_len, self.hidden_dim])
        ht = x.new_zeros([batch_size, self.hidden_dim])
        feature_x = self.feature_extra_x(x)

        for t in range(seq_len):
            xt = feature_x[:, t, :]
            zt, zt_loc, zt_logvar = self.encode(xt, ht)
            yt = self.decode(zt, ht)

            h[:, t, :] = ht
            self.z[:, t, :] = zt
            self.z_loc[:, t, :] = zt_loc
            self.z_logvar[:, t, :] = zt_logvar
            y[:, t, :] = yt
            ht = self.recurrence(xt, zt, ht)

        self.z_loc_prior, self.z_logvar_prior = self.theta_norm_1(h, return_logvar=True)
        return y

    def compute_loss(self, x, y, anneling_factor=1.0):
        b, l, _ = x.shape

        recon = functional.mse_loss(x, y, reduction="mean")
        kld = -0.5 * torch.sum(
            self.z_logvar - self.z_logvar_prior - torch.div(
                (self.z_logvar.exp() + (self.z_loc - self.z_loc_prior).pow(2)),
                self.z_logvar_prior.exp() + 1e-10))

        return recon + kld * anneling_factor / (b * l)
