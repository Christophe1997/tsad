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


@register("vrnn", "n_features", "hidden_dim", "z_dim", "dropout", "phi_dense", "theta_dense")
class VRNN(nn.Module):
    """Original paper: A Recurrent Latent Variable Model for Sequential Data (https://arxiv.org/abs/1506.02216)
    """

    def __init__(self, n_features=1, hidden_dim=256, z_dim=4, dropout=0.1, phi_dense=False, theta_dense=False):
        super(VRNN, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_features = n_features

        # inference, share same rnn with generation
        self.phi_rnn = nn.GRUCell(n_features + z_dim, hidden_dim)
        self.phi_p_z_x = NormalParam(hidden_dim + n_features, z_dim)
        if phi_dense:
            self.phi_dense = MLPEmbedding(n_features + hidden_dim, n_features + hidden_dim, [n_features + hidden_dim],
                                          dropout, activate=nn.ReLU())
        else:
            self.phi_dense = nn.Identity()
        # generation
        self.theta_p_z = NormalParam(hidden_dim, z_dim)
        self.theta_p_x_z = NormalParam(z_dim + hidden_dim, n_features)
        if theta_dense:
            self.theta_dense = MLPEmbedding(z_dim + hidden_dim, z_dim + hidden_dim, [z_dim + hidden_dim], dropout,
                                            activate=nn.ReLU())
        else:
            self.theta_dense = nn.Identity()

    def inference(self, x, h):
        x_with_h = torch.cat([h, x], dim=1)
        x_with_h = self.phi_dense(x_with_h)
        z_loc, z_scale, z_dist = self.phi_p_z_x(x_with_h, return_dist=True)
        return z_loc, z_scale, z_dist

    def generate_x(self, z, h):
        z_with_h = torch.cat([z, h], dim=1)
        z_with_h = self.theta_dense(z_with_h)
        x_loc, x_scale, x_dist = self.theta_p_x_z(z_with_h, return_dist=True)
        return x_loc, x_scale, x_dist

    def recurrence(self, feature_x, featurn_z, h):
        feature_x_with_z = torch.cat([feature_x, featurn_z], dim=1)
        h = self.phi_rnn(feature_x_with_z, h)
        return h

    def forward(self, x, return_prob=False, return_loss=True, n_sample=1):
        batch_size, seq_len, _ = x.shape
        y_loc = x.new_zeros([batch_size, seq_len, self.n_features])
        y_scale = x.new_zeros([batch_size, seq_len, self.n_features])
        ht = x.new_zeros([batch_size, self.hidden_dim])

        recon = 0
        kld = 0

        for t in range(seq_len):
            xt = x[:, t, :]
            zt_loc_prior, zt_scale_prior, zt_prior_dist = self.theta_p_z(ht, return_dist=True)
            zt_loc, zt_scale, zt_dist = self.inference(xt, ht)
            zt = zt_dist.rsample([n_sample]).mean(0)
            yt_loc, yt_scale, yt_dist = self.generate_x(zt, ht)

            recon += -yt_dist.log_prob(x[:, t, :]).sum()
            kld += tdist.kl_divergence(zt_dist, zt_prior_dist).sum()

            y_loc[:, t, :] = yt_loc
            y_scale[:, t, :] = yt_scale
            ht = self.recurrence(xt, zt, ht)

        first_term = y_loc if not return_prob else (y_loc, y_scale)
        return first_term if not return_loss else (first_term, (recon, kld))


@register("omni", "n_features", "hidden_dim", "z_dim", "dropout", "phi_dense", "theta_dense")
class Omni(nn.Module):

    def __init__(self, n_features=1, hidden_dim=500, z_dim=4, dropout=0.1, phi_dense=False, theta_dense=False):
        super(Omni, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_features = n_features

        # inference
        self.phi_rnn = nn.GRUCell(n_features, hidden_dim)
        self.phi_p_z_x = NormalParam(hidden_dim + z_dim, z_dim)
        if phi_dense:
            self.phi_dense = MLPEmbedding(hidden_dim + z_dim, hidden_dim + z_dim, [hidden_dim + z_dim], dropout=dropout,
                                          activate=nn.ReLU())
        else:
            self.phi_dense = nn.Identity()

        # generation
        self.theta_rnn = nn.GRUCell(z_dim, hidden_dim)
        if theta_dense:
            self.theta_dense = MLPEmbedding(hidden_dim, hidden_dim, [hidden_dim], dropout=dropout, activate=nn.ReLU())
        else:
            self.theta_dense = nn.Identity()
        self.theta_p_x_z = NormalParam(hidden_dim, n_features)

    def generate_x(self, h):
        h = self.theta_dense(h)
        x_loc, x_scale, x_dist = self.theta_p_x_z(h, return_dist=True)
        return x_loc, x_scale, x_dist

    def inference(self, h, z_prev):
        h_with_zt = torch.cat([h, z_prev], dim=1)
        h_with_zt = self.phi_dense(h_with_zt)
        zt_loc, zt_scale, zt_dist = self.phi_p_z_x(h_with_zt, return_dist=True)

        return zt_loc, zt_scale, zt_dist

    def generate_z(self, z_prev):
        z_loc, z_scale = z_prev, torch.ones_like(z_prev)
        return z_loc, z_scale, dist.Normal(z_loc, z_scale).to_event(1)

    def forward(self, x, return_prob=False, return_loss=True, n_sample=1):
        b, l, _ = x.shape
        phi_h = x.new_zeros([b, self.hidden_dim])
        theta_h = x.new_zeros([b, self.hidden_dim])
        zt = x.new_zeros([b, self.z_dim])
        y_loc = torch.zeros_like(x)
        y_scale = torch.zeros_like(x)
        z_prior_dist = dist.Normal(x.new_zeros([b, self.z_dim]), x.new_ones([b, self.z_dim])).to_event(1)

        recon = 0
        kld = 0
        for t in range(l):
            xt = x[:, t, :]
            phi_h = self.phi_rnn(xt, phi_h)
            _, _, zt_dist = self.inference(phi_h, zt)
            zt = zt_dist.rsample([n_sample]).mean(0)
            theta_h = self.theta_rnn(zt, theta_h)
            yt_loc, yt_scale, yt_dist = self.generate_x(theta_h)
            y_loc[:, t, :] = yt_loc
            y_scale[:, t, :] = yt_scale

            recon += -yt_dist.log_prob(xt).sum()
            kld += tdist.kl_divergence(zt_dist, z_prior_dist).sum()
            _, _, z_prior_dist = self.generate_z(zt)

        first_term = y_loc if not return_prob else (y_loc, y_scale)
        return first_term if not return_loss else (first_term, (recon, kld))


@register("rvae", "n_features", "hidden_dim", "z_dim", "dropout", "phi_dense", "theta_dense")
class RVAE(nn.Module):
    """Original paper: A Recurrent Variational Autoencoder for Speech Enhancement (https://arxiv.org/abs/1910.10942)
    """

    def __init__(self, n_features=1, hidden_dim=256, z_dim=4, dropout=0.1, phi_dense=False, theta_dense=False):
        super(RVAE, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        # inference, share the rnn_2 with generation
        self.phi_rnn_1 = nn.GRU(n_features, hidden_dim, batch_first=True)
        self.phi_rnn_2 = nn.GRUCell(z_dim, hidden_dim)
        self.phi_p_z_x = NormalParam(hidden_dim + hidden_dim, z_dim)
        if phi_dense:
            self.phi_dense = MLPEmbedding(hidden_dim + hidden_dim, hidden_dim + hidden_dim, [hidden_dim + hidden_dim],
                                          dropout, activate=nn.ReLU())
        else:
            self.phi_dense = nn.Identity()

        # generation
        self.theta_rnn = nn.GRUCell(z_dim, hidden_dim)
        self.theta_p_x_z = NormalParam(hidden_dim, n_features)
        if theta_dense:
            self.theta_dense = MLPEmbedding(hidden_dim, hidden_dim, [hidden_dim], dropout, activate=nn.ReLU())
        else:
            self.theta_dense = nn.Identity()

    def inference(self, zh, xh_r):
        zh_with_xh_r = torch.cat([zh, xh_r], dim=-1)
        zh_with_xh_r = self.phi_dense(zh_with_xh_r)
        z_loc, z_scale, z_dist = self.phi_p_z_x(zh_with_xh_r, return_dist=True)
        return z_loc, z_scale, z_dist

    def generate_x(self, h):
        h = self.theta_dense(h)
        x_loc, x_scale, x_dist = self.theta_p_x_z(h, return_dist=True)
        return x_loc, x_scale, x_dist

    def forward(self, x, return_prob=False, return_loss=True, n_sample=1):
        b, l, _ = x.shape
        phi_zh = x.new_zeros([b, self.hidden_dim])
        theta_zh = x.new_zeros([b, self.hidden_dim])
        zt = x.new_zeros([b, self.z_dim])
        y_loc = torch.zeros_like(x)
        y_scale = torch.zeros_like(x)
        prior_z = dist.Normal(x.new_zeros([b, self.z_dim]), x.new_ones([b, self.z_dim])).to_event(1)

        recon = 0
        kld = 0

        x_reverse = torch.flip(x, [1])
        xh, _ = self.phi_rnn_1(x_reverse)
        xh_r = torch.flip(xh, [1])

        for t in range(l):
            phi_zh = self.phi_rnn_2(zt, phi_zh)
            z_loc, z_scale, z_dist = self.inference(phi_zh, xh_r[:, t, :])
            zt = z_dist.rsample([n_sample]).mean(0)
            theta_zh = self.theta_rnn(zt, theta_zh)
            yt_loc, yt_scale, yt_dist = self.generate_x(theta_zh)

            recon += -yt_dist.log_prob(x[:, t, :]).sum()
            kld += tdist.kl_divergence(z_dist, prior_z).sum()

            y_loc[:, t, :] = yt_loc
            y_scale[:, t, :] = yt_scale

        first_term = y_loc if not return_prob else (y_loc, y_scale)
        return first_term if not return_loss else (first_term, (recon, kld))


@register("srnn", "n_features", "hidden_dim", "z_dim", "dropout", "phi_dense", "theta_dense")
class SRNN(nn.Module):
    """Original paper: Sequential Neural Models with Stochastic Layers (https://arxiv.org/abs/1605.07571)
    """

    def __init__(self, n_features=1, hidden_dim=256, z_dim=4, dropout=0.1, phi_dense=False, theta_dense=False):
        super(SRNN, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        # inference, share the rnn_1 with generation
        self.phi_rnn_1 = nn.GRU(n_features, hidden_dim, batch_first=True)
        self.phi_rnn_2 = nn.GRU(n_features + hidden_dim, hidden_dim, batch_first=True)
        self.phi_p_z_x = NormalParam(z_dim + hidden_dim, z_dim)
        if phi_dense:
            self.phi_dense = MLPEmbedding(z_dim + hidden_dim, z_dim + hidden_dim, [z_dim + hidden_dim], dropout,
                                          activate=nn.ReLU())
        else:
            self.phi_dense = nn.Identity()

        # generation
        self.theta_p_z = NormalParam(z_dim + hidden_dim, z_dim)
        self.theta_p_x_z = NormalParam(z_dim + hidden_dim, n_features)
        if theta_dense:
            self.theta_dense = MLPEmbedding(z_dim + hidden_dim, z_dim + hidden_dim, [z_dim + hidden_dim], dropout,
                                            activate=nn.ReLU())
        else:
            self.theta_dense = nn.Identity()

    def inference(self, x, h, n_sample=1):
        b, l, _ = x.shape
        z_loc = x.new_zeros([b, l, self.z_dim])
        z_scale = x.new_zeros([b, l, self.z_dim])
        z = x.new_zeros([b, l, self.z_dim])
        zt = x.new_zeros([b, self.z_dim])

        x_with_h = torch.cat([x, h], dim=-1)
        h, _ = self.phi_rnn_2(torch.flip(x_with_h, [1]))
        h = torch.flip(h, [1])

        for t in range(l):
            zt_with_h = torch.cat([zt, h[:, t, :]], dim=-1)
            zt_with_h = self.phi_dense(zt_with_h)
            zt_loc, zt_scale, zt_dist = self.phi_p_z_x(zt_with_h, return_dist=True)
            zt = zt_dist.rsample([n_sample]).mean(0)

            z_loc[:, t, :] = zt_loc
            z_scale[:, t, :] = zt_scale
            z[:, t, :] = zt

        return z_loc, z_scale, z

    def generate_x(self, z, h):
        z_with_h = torch.cat([z, h], dim=-1)
        z_with_h = self.theta_dense(z_with_h)
        x_loc, x_scale, x_dist = self.theta_p_x_z(z_with_h, return_dist=True)
        return x_loc, x_scale, x_dist

    def generate_z(self, h, z_lag):
        z_with_h = torch.cat([h, z_lag], dim=-1)
        z_loc, z_scale, z_dist = self.theta_p_z(z_with_h, return_dist=True)
        return z_loc, z_scale, z_dist

    def forward(self, x, return_prob=False, return_loss=True, n_sample=1):
        b, l, _ = x.shape
        h, _ = self.phi_rnn_1(lag(x))
        z_loc, z_scale, z = self.inference(x, h, n_sample=n_sample)
        z_dist = dist.Normal(z_loc, z_scale).to_event(1)
        y_loc, y_scale, y_dist = self.generate_x(z, h)
        z_prior_loc, z_prior_scale, z_dist_prior = self.generate_z(h, lag(z))

        recon = -y_dist.log_prob(x).sum()
        kld = tdist.kl_divergence(z_dist, z_dist_prior).sum()

        first_term = y_loc if not return_prob else (y_loc, y_scale)
        return first_term if not return_loss else (first_term, (recon, kld))


@register("tfvae", "n_features", "z_dim", "nhead", "nlayers", "phi_dense", "theta_dense", "dropout",
          d_model="hidden_dim",
          dim_feedforward="dense_dim")
class TransformerVAE(nn.Module):

    def __init__(self, n_features=1, d_model=256, z_dim=4, nhead=8, nlayers=6,
                 dim_feedforward=1024, dropout=0.1, phi_dense=False, theta_dense=False):
        super(TransformerVAE, self).__init__()
        self.z_dim = z_dim
        self.d_model = d_model
        self.n_features = n_features

        # inference, share the transformer encoder with generation
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

    def inference(self, x, n_sample=1):
        b, l, _ = x.shape
        h = self.encode(x)
        h = self.phi_dense(h)
        z_loc = x.new_zeros([b, l, self.z_dim])
        z_scale = x.new_zeros([b, l, self.z_dim])
        z = x.new_zeros([b, l, self.z_dim])
        zt = x.new_zeros([b, self.z_dim])
        for t in range(l):
            zt_with_h = torch.cat([zt, h[:, t, :]], dim=-1)
            zt_loc, zt_scale, zt_dist = self.phi_p_z_x(zt_with_h, return_dist=True)
            zt = zt_dist.rsample([n_sample]).mean(0)

            z_loc[:, t, :] = zt_loc
            z_scale[:, t, :] = zt_scale
            z[:, t, :] = zt

        return z_loc, z_scale, z

    def generate_z(self, h, z_lag):
        z_with_h = torch.cat([h, z_lag], dim=-1)
        z_loc, z_scale, z_dist = self.theta_p_z(z_with_h, return_dist=True)
        return z_loc, z_scale, z_dist

    def generate_x(self, z, h):
        z_with_h = torch.cat([z, h], dim=-1)
        x_loc, x_scale, x_dist = self.theta_p_x_z(z_with_h, return_dist=True)
        return x_loc, x_scale, x_dist

    def forward(self, x, return_prob=False, return_loss=True, n_sample=1):
        b, l, _ = x.shape

        z_loc, z_scale, z = self.inference(x, n_sample=n_sample)
        z_dist = dist.Normal(z_loc, z_scale).to_event(1)

        h = self.encode(lag(x))
        h = self.theta_dense(h)
        y_loc, y_scale, y_dist = self.generate_x(z, h)

        z_prior_loc, z_prior_scale, z_dist_prior = self.generate_z(h, lag(z))

        recon = -y_dist.log_prob(x).sum()
        kld = tdist.kl_divergence(z_dist, z_dist_prior).sum()

        first_term = y_loc if not return_prob else (y_loc, y_scale)
        return first_term if not return_loss else (first_term, (recon, kld))
