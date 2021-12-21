import numpy as np
import torch
from torch import nn

from tsad.models.submodule import NormalParam, MLPEmbedding


def reparameterization(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return torch.addcmul(mean, eps, std)


def kld_gaussian(mean1, mean2, logvar1, logvar2):
    kld_elem = logvar2 - logvar1 + torch.div(logvar1.exp() + (mean1 - mean2) ** 2, logvar2.exp()) - 1
    return torch.sum(0.5 * kld_elem)


def nll_gaussian(mean, logvar, x):
    nll_elem = logvar + torch.div((x - mean) ** 2, logvar.exp()) + np.log(2 * np.pi)
    return torch.sum(0.5 * nll_elem)


class VRNN(nn.Module):
    """ Original paper: A Recurrent Latent Variable Model for Sequential Data (https://arxiv.org/abs/1506.02216)
    """

    def __init__(self, n_features=1, hidden_dim=256, z_dim=4, dropout=0.1):
        super(VRNN, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.dim_feature_x = 64
        self.dim_feature_z = 32

        self.rnn = nn.GRUCell(input_size=self.dim_feature_x + z_dim, hidden_size=hidden_dim)
        self.phi_norm = NormalParam(hidden_dim + self.dim_feature_x, z_dim)
        self.theta_norm_1 = NormalParam(hidden_dim, z_dim)
        self.theta_norm_2 = NormalParam(self.dim_feature_z + hidden_dim, n_features)

        self.feature_extra_x = MLPEmbedding(n_features, self.dim_feature_x, dropout=dropout)
        self.feature_extra_z = MLPEmbedding(z_dim, self.dim_feature_z, [self.dim_feature_z // 2], dropout=dropout)
        self.h0 = nn.Parameter(torch.zeros(hidden_dim))

    def encode(self, x, h):
        h_with_x = torch.cat([h, x], dim=1)
        z_mean, z_logvar = self.phi_norm(h_with_x, return_logvar=True)
        z = reparameterization(z_mean, z_logvar)
        return z, z_mean, z_logvar

    def decode(self, z, h, return_prob=False):
        feature_z = self.feature_extra_z(z)
        z_with_h = torch.cat([feature_z, h], dim=1)
        x_mean, x_logvar = self.theta_norm_2(z_with_h, return_logvar=True)
        return x_mean if not return_prob else (x_mean, x_logvar)

    def recurrence(self, x, z, h):
        x_with_z = torch.cat([x, z], dim=1)
        h = self.rnn(x_with_z, h)
        return h

    def forward(self, x, return_prob=False, return_loss=True):
        batch_size, seq_len, _ = x.shape
        z_mean = x.new_zeros([batch_size, seq_len, self.z_dim])
        z_logvar = x.new_zeros([batch_size, seq_len, self.z_dim])
        y_mean = x.new_zeros([batch_size, seq_len, self.n_features])
        y_logvar = x.new_zeros([batch_size, seq_len, self.n_features])
        z_mean_prior = x.new_zeros([batch_size, seq_len, self.z_dim])
        z_logvar_prior = x.new_zeros([batch_size, seq_len, self.z_dim])
        ht = self.h0.expand([batch_size, self.hidden_dim])
        feature_x = self.feature_extra_x(x)

        for t in range(seq_len):
            xt = feature_x[:, t, :]
            zt_mean_prior, zt_logvar_prior = self.theta_norm_1(ht, return_logvar=True)
            zt, zt_mean, zt_logvar = self.encode(xt, ht)
            yt_mean, yt_logvar = self.decode(zt, ht, return_prob=True)

            z_mean[:, t, :] = zt_mean
            z_logvar[:, t, :] = zt_logvar
            z_mean_prior[:, t, :] = zt_mean_prior
            z_logvar_prior[:, t, :] = zt_logvar_prior
            y_mean[:, t, :] = yt_mean
            y_logvar[:, t, :] = yt_logvar
            ht = self.recurrence(xt, zt, ht)

        recon = nll_gaussian(y_mean, y_logvar, x)
        kld = kld_gaussian(z_mean, z_mean_prior, z_logvar, z_logvar_prior)
        first_term = y_mean if not return_prob else (y_mean, y_logvar)
        return first_term if not return_loss else (first_term, (recon, kld))
