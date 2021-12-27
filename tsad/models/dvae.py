import numpy as np
import torch
from torch import nn
import pyro.distributions as dist
import torch.distributions as tdist

from tsad.config import register
from tsad.models.submodule import NormalParam, MLPEmbedding, PositionalEncoding, Conv1DEmbedding


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


def lag(x):
    # [B, L, N]
    b, _, n = x.shape
    x0 = x.new_zeros([b, 1, n])
    return torch.cat([x0, x[:, :-1, :]], dim=1)


@register("vrnn", "n_features", "hidden_dim", "z_dim" "dropout", "feature_x", "feature_z")
class VRNN(nn.Module):
    """ Original paper: A Recurrent Latent Variable Model for Sequential Data (https://arxiv.org/abs/1506.02216)
    """

    def __init__(self, n_features=1, hidden_dim=256, z_dim=4, dropout=0.1, feature_x=64, feature_z=32):
        super(VRNN, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_features = n_features

        # embedding
        if feature_x is None:
            self.dim_feature_x = n_features
            self.feature_extra_x = nn.Identity()
        else:
            self.dim_feature_x = feature_x
            self.feature_extra_x = MLPEmbedding(n_features, self.dim_feature_x, dropout=dropout)

        if feature_z is None:
            self.dim_feature_z = z_dim
            self.feature_extra_z = nn.Identity()
        else:
            self.dim_feature_z = feature_z
            self.feature_extra_z = MLPEmbedding(z_dim, self.dim_feature_z, [self.dim_feature_z // 2], dropout=dropout)

        # share same rnn
        self.rnn = nn.GRUCell(input_size=self.dim_feature_x + self.dim_feature_z, hidden_size=hidden_dim)
        # inference
        self.phi_p_z_x = NormalParam(hidden_dim + self.dim_feature_x, z_dim)
        # generation
        self.theta_prior_z = NormalParam(hidden_dim, z_dim)
        self.theta_p_x_z = NormalParam(self.dim_feature_z + hidden_dim, n_features)

    def inference(self, feature_x, h):
        feature_x_with_h = torch.cat([h, feature_x], dim=1)
        z_loc, z_scale, z_dist = self.phi_p_z_x(feature_x_with_h, return_dist=True)
        return z_loc, z_scale, z_dist

    def generate_x(self, feature_z, h):
        feature_z_with_h = torch.cat([feature_z, h], dim=1)
        x_loc, x_scale, x_dist = self.theta_p_x_z(feature_z_with_h, return_dist=True)
        return x_loc, x_scale, x_dist

    def recurrence(self, feature_x, featurn_z, h):
        feature_x_with_z = torch.cat([feature_x, featurn_z], dim=1)
        h = self.rnn(feature_x_with_z, h)
        return h

    def forward(self, x, return_prob=False, return_loss=True):
        batch_size, seq_len, _ = x.shape
        y_loc = x.new_zeros([batch_size, seq_len, self.n_features])
        y_scale = x.new_zeros([batch_size, seq_len, self.n_features])
        ht = x.new_zeros([batch_size, self.hidden_dim])
        feature_x = self.feature_extra_x(x)

        recon = 0
        kld = 0

        for t in range(seq_len):
            feature_xt = feature_x[:, t, :]
            zt_loc_prior, zt_scale_prior, zt_prior_dist = self.theta_prior_z(ht, return_dist=True)
            zt_loc, zt_scale, zt_dist = self.inference(feature_xt, ht)
            zt = zt_dist.rsample()
            feature_zt = self.feature_extra_z(zt)
            yt_loc, yt_scale, yt_dist = self.generate_x(feature_zt, ht)

            recon += -yt_dist.log_prob(x[:, t, :]).sum()
            kld += tdist.kl_divergence(zt_dist, zt_prior_dist).sum()

            y_loc[:, t, :] = yt_loc
            y_scale[:, t, :] = yt_scale
            ht = self.recurrence(feature_xt, feature_zt, ht)

        first_term = y_loc if not return_prob else (y_loc, y_scale)
        return first_term if not return_loss else (first_term, (recon, kld))


class Omni(nn.Module):

    def __init__(self, n_features=1, hidden_dim=500, z_dim=4, dropout=0.1, dense_dim=None):
        super(Omni, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_features = n_features

        if dense_dim is None:
            dense_dim = hidden_dim

        # inference
        self.phi_rnn = nn.GRUCell(n_features, hidden_dim)
        self.phi_dense = MLPEmbedding(hidden_dim + z_dim, dense_dim, [dense_dim], dropout=dropout, activate=nn.ReLU())
        self.phi_p_z_x = NormalParam(hidden_dim, z_dim)

        # generation
        self.theta_rnn = nn.GRUCell(z_dim, hidden_dim)
        self.theta_dense = MLPEmbedding(hidden_dim, dense_dim, [dense_dim], dropout=dropout, activate=nn.ReLU())
        self.theta_p_x_z = NormalParam(hidden_dim, n_features)

    def generate(self, h):
        h = self.theta_dense(h)
        x_loc, x_scale, x_dist = self.theta_p_x_z(h, return_dist=True)
        return x_loc, x_scale, x_dist

    def inference(self, h, z_prev):
        h_with_zt = torch.cat([h, z_prev], dim=1)
        h_with_zt = self.phi_dense(h_with_zt)
        zt_loc, zt_scale, zt_dist = self.phi_p_z_x(h_with_zt, return_dist=True)

        return zt_loc, zt_scale, zt_dist

    def forward(self, x, return_prob=False, return_loss=True):
        b, l, _ = x.shape
        phi_h = x.new_zeros([b, self.hidden_dim])
        theta_h = x.new_zeros([b, self.hidden_dim])
        zt = x.new_zeros([b, self.z_dim])
        y_loc = x.new_zeros([b, l, self.n_features])
        y_scale = x.new_zeros([b, l, self.n_features])
        z_prior_dist = dist.Normal(x.new_zeros([b, self.z_dim]), x.new_ones([b, self.z_dim])).to_event(1)

        recon = 0
        kld = 0
        for t in range(l):
            xt = x[:, t, :]
            phi_h = self.phi_rnn(xt, phi_h)
            _, _, zt_dist = self.inference(phi_h, zt)
            zt = zt_dist.rsample()
            theta_h = self.theta_rnn(zt, theta_h)
            yt_loc, yt_scale, yt_dist = self.generate(theta_h)
            y_loc[:, t, :] = yt_loc
            y_scale[:, t, :] = yt_scale

            z_prior = z_prior_dist.rsample()
            recon += -yt_dist.log_prob(xt).sum()
            kld += (tdist.kl_divergence(zt_dist, z_prior_dist)).sum()
            z_prior_dist = dist.Normal(z_prior, x.new_ones([b, self.z_dim])).to_event(1)

        first_term = y_loc if not return_prob else (y_loc, y_scale)
        return first_term if not return_loss else (first_term, (recon, kld))


@register("tfvae", "n_features", "z_dim", "nhead", "nlayers", "phi_dense", "theta_dense",
          d_model="hidden_dim",
          dim_feedforward="dim_dense")
class TransformerVAE(nn.Module):

    def __init__(self, n_features=1, d_model=256, z_dim=4, nhead=8, nlayers=6,
                 dim_feedforward=1024, dropout=0.1, phi_dense=False, theta_dense=False):
        super(TransformerVAE, self).__init__()
        self.z_dim = z_dim
        self.d_model = d_model
        self.n_features = n_features

        # inference, share the same transformer encoder wth generation
        self.phi_pos_encoder = PositionalEncoding(d_model, batch_first=True)
        self.phi_x_embedding = Conv1DEmbedding(n_features, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.phi_transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        if phi_dense:
            self.phi_dense = MLPEmbedding(d_model, d_model, dropout=dropout, activate=nn.ReLU())
        else:
            self.phi_dense = nn.Identity()
        self.phi_p_z_x = NormalParam(d_model + z_dim, z_dim)

        # generation
        self.theta_p_x_z = NormalParam(d_model + z_dim, n_features)
        self.theta_p_z = NormalParam(d_model + z_dim, z_dim)

        if theta_dense:
            self.theta_dense = MLPEmbedding(d_model, d_model, dropout=dropout, activate=nn.ReLU())
        else:
            self.theta_dense = nn.Identity()

    def encode(self, x):
        embedding = self.phi_x_embedding(x) + self.phi_pos_encoder(x)
        mask = torch.triu(x.new_full((x.size(1), x.size(1)), float('-inf')), diagonal=1)
        return self.phi_transformer_encoder(embedding, mask=mask)

    def inference(self, x):
        b, l, _ = x.shape
        h = self.encode(x)
        h = self.phi_dense(h)
        z_loc = x.new_zeros([b, l, self.z_dim])
        z_scale = x.new_zeros([b, l, self.z_dim])
        zt = x.new_zeros([b, self.z_dim])
        for t in range(l):
            zt_with_h = torch.cat([zt, h[:, t, :]], dim=-1)
            zt_loc, zt_scale = self.phi_p_z_x(zt_with_h)
            z_loc[:, t, :] = zt_loc
            z_scale[:, t, :] = zt_scale
        return z_loc, z_scale

    def generate_z(self, h, z_lag):
        z_with_h = torch.cat([h, z_lag], dim=-1)
        z_loc, z_scale = self.theta_p_z(z_with_h)
        return z_loc, z_scale

    def generate_x(self, z, h):
        z_with_h = torch.cat([z, h], dim=-1)
        x_loc, x_scale = self.theta_p_x_z(z_with_h)
        return x_loc, x_scale

    def forward(self, x, return_prob=False, return_loss=True):
        b, l, _ = x.shape
        x_lag = lag(x)

        z_loc, z_scale = self.inference(x)
        z_dist = dist.Normal(z_loc, z_scale).to_event(1)
        z = z_dist.rsample()

        h = self.encode(x_lag)
        h = self.theta_dense(h)
        y_loc, y_scale = self.generate_x(z, h)
        y_dist = dist.Normal(y_loc, y_scale).to_event(1)

        z_lag = lag(z)
        z_prior_loc, z_prior_scale = self.generate_z(h, z_lag)
        z_dist_prior = dist.Normal(z_prior_loc, z_prior_scale).to_event(1)

        recon = -y_dist.log_prob(x).sum()
        kld = tdist.kl_divergence(z_dist, z_dist_prior)

        first_term = y_loc if not return_prob else (y_loc, y_scale)
        return first_term if not return_loss else (first_term, (recon, kld))
