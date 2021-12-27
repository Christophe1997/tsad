import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from torch import nn

from tsad.config import register
from tsad.models import dvae
from tsad.models.submodule import NormalParam, MLPEmbedding, Conv1DEmbedding, PositionalEncoding


@register("vrnn_pyro", "n_features", "hidden_dim", "dropout", "feature_x", "feature_z", z_dim="emb_dim")
class VRNNPyro(dvae.VRNN):

    def __init__(self, *args, **kwargs):
        super(VRNNPyro, self).__init__(*args, **kwargs)

    def model(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("vrnn", self)
        h = x.new_zeros([b, self.hidden_dim])

        feature_x = self.feature_extra_x(x)
        with pyro.plate("data", b):
            for t in range(1, l + 1):
                z_loc, z_scale = self.theta_prior_z(h)
                with poutine.scale(None, annealing_factor):
                    zt = pyro.sample(f"z_{t}", dist.Normal(z_loc, z_scale).to_event(1))

                zt = self.feature_extra_z(zt)
                x_loc, x_scale, x_dist = self.generate_x(zt, h)
                pyro.sample(f"obs_{t}", x_dist, obs=x[:, t - 1, :])
                h = self.recurrence(feature_x[:, t - 1, :], zt, h)

    def guide(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("vrnn", self)
        h = x.new_zeros([b, self.hidden_dim])

        feature_x = self.feature_extra_x(x)
        with pyro.plate("data", b):
            for t in range(1, l + 1):
                xt = feature_x[:, t - 1, :]
                z_loc, z_scale, zdist = self.inference(xt, h)
                with poutine.scale(None, annealing_factor):
                    zt = pyro.sample(f"z_{t}", zdist)

                zt = self.feature_extra_z(zt)
                h = self.recurrence(xt, zt, h)


class RVAE(nn.Module):
    """Original paper: A Recurrent Variational Autoencoder for Speech Enhancement (https://arxiv.org/abs/1910.10942)

    It's an implementation based on pyro.
    """

    def __init__(self, n_features=1, hidden_dim=256, z_dim=4, dropout=0.1, feature_x=64, feature_z=32, n_sample=1):
        super(RVAE, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_sample = n_sample

        # embedding
        if feature_x is None:
            self.dim_feature_x = n_features
            self.feature_extra_x = nn.Identity()
        else:
            self.dim_feature_x = feature_x
            self.feature_extra_x = MLPEmbedding(n_features, feature_x, dropout=dropout)

        if feature_z is None:
            self.dim_feature_z = z_dim
            self.feature_extra_z = nn.Identity()
        else:
            self.dim_feature_z = feature_z
            self.feature_extra_z = MLPEmbedding(z_dim, feature_z, [self.dim_feature_z // 2], dropout=dropout)

        # generation
        self.theta_rnn = nn.GRU(self.dim_feature_z, hidden_dim, batch_first=True)
        self.theta_norm = NormalParam(hidden_dim, n_features)

        # inference
        self.phi_rnn_1 = nn.GRU(self.dim_feature_x, hidden_dim, batch_first=True)
        self.phi_rnn_2 = nn.GRUCell(self.dim_feature_z, hidden_dim)
        self.phi_norm = NormalParam(hidden_dim + hidden_dim, z_dim)

    def model(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("rvae", self)
        prior_z = dist.Normal(x.new_zeros([b, self.z_dim]), x.new_ones([b, self.z_dim])).to_event(1)
        h = x.new_zeros([1, b, self.hidden_dim])
        with pyro.plate("data", b):
            for t in range(1, l + 1):
                with poutine.scale(None, annealing_factor):
                    z_t = pyro.sample(f"z_{t}", prior_z)
                z_t = self.feature_extra_z(z_t)
                _, h = self.theta_rnn(z_t.unsqueeze(1), h)
                x_loc, x_scale = self.theta_norm(h.squeeze())
                pyro.sample(f"obs_{t}", dist.Normal(x_loc, x_scale).to_event(1), obs=x[:, t - 1, :])

    def guide(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("rvae", self)
        x = self.feature_extra_x(x)
        x_reverse = torch.flip(x, [1])
        xh, _ = self.phi_rnn_1(x_reverse)
        xh_r = torch.flip(xh, [1])
        zh = x.new_zeros([b, self.hidden_dim])
        z_t = x.new_zeros([b, self.dim_feature_z])
        with pyro.plate("data", b):
            for t in range(1, l + 1):
                zh = self.phi_rnn_2(z_t, zh)
                zh_with_xh_r = torch.cat([zh, xh_r[:, t - 1, :]], dim=1)
                z_loc, z_scale = self.phi_norm(zh_with_xh_r)
                with poutine.scale(None, annealing_factor):
                    z_t = pyro.sample(f"z_{t}", dist.Normal(z_loc, z_scale).to_event(1))
                z_t = self.feature_extra_z(z_t)

    def encode(self, x):
        b, l, _ = x.shape
        x = self.feature_extra_x(x)
        x_reverse = torch.flip(x, [1])
        xh, _ = self.phi_rnn_1(x_reverse)
        xh_r = torch.flip(xh, [1])
        zh = x.new_zeros([b, self.hidden_dim])
        z = x.new_zeros([b, l, self.z_dim])
        z_t = x.new_zeros([b, self.dim_feature_z])

        for t in range(l):
            zh = self.phi_rnn_2(z_t, zh)
            zh_with_xh_r = torch.cat([zh, xh_r[:, t, :]], dim=1)
            z_loc, z_scale = self.phi_norm(zh_with_xh_r)
            z_t = dist.Normal(z_loc, z_scale).to_event(1).rsample([self.n_sample])
            z_t = z_t.mean(dim=0)
            z[:, t, :] = z_t
            z_t = self.feature_extra_z(z_t)

        return z

    def decode(self, z):
        b, l, _ = z.shape
        z = self.feature_extra_z(z)
        h, _ = self.theta_rnn(z)
        x_loc, x_scale = self.theta_norm(h)
        return x_loc, x_scale

    def forward(self, x, return_prob=False):
        z = self.encode(x)
        x_loc, x_scale = self.decode(z)
        return x_loc if not return_prob else (x_loc, x_scale)


@register("tfvae", "n_features", "z_dim", "nhead", "nlayers", "phi_dense", "theta_dense",
          d_model="hidden_dim",
          dim_feedforward="dim_dense")
class TransformerVAEPyro(dvae.TransformerVAE):

    def __init__(self, *args, **kwargs):
        super(TransformerVAEPyro, self).__init__(*args, **kwargs)

    def model(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("tfvae", self)
        x_lag = dvae.lag(x)
        h = self.encode(x_lag)
        h = self.theta_dense(h)
        zt = x.new_zeros([b, self.z_dim])

        with pyro.plate("data", b):
            for t in range(1, l + 1):
                ht = h[:, t - 1, :]
                zt_loc, zt_scale = self.generate_z(ht, zt)
                with poutine.scale(None, annealing_factor):
                    zt = pyro.sample(f"z_{t}", dist.Normal(zt_loc, zt_scale).to_event(1))
                x_loc, x_scale = self.generate_x(zt, ht)
                pyro.sample(f"obs_{t}", dist.Normal(x_loc, x_scale).to_event(1))

    def guide(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        h = self.encode(x)
        h = self.phi_dense(h)
        zt = x.new_zeros([b, self.z_dim])

        with pyro.plate("data", b):
            for t in range(1, l + 1):
                zt_with_h = torch.cat([zt, h[:, t - 1, :]], dim=-1)
                zt_loc, zt_scale = self.phi_p_z_x(zt_with_h)
                with poutine.scale(None, annealing_factor):
                    zt = pyro.sample(f"z_{t}", dist.Normal(zt_loc, zt_scale).to_event(1))
