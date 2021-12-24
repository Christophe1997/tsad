import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from torch import nn

from tsad.models.submodule import NormalParam, MLPEmbedding, Conv1DEmbedding, TimePositionalEncoding


class VRNN(nn.Module):
    """Original paper: A Recurrent Latent Variable Model for Sequential Data (https://arxiv.org/abs/1506.02216)

    It's an implementation based on pyro.
    """

    def __init__(self, n_features=1, hidden_dim=256, z_dim=4, dropout=0.1, feature_x=64, feature_z=32, n_sample=1):
        super(VRNN, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_sample = n_sample

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
        self.phi_norm = NormalParam(hidden_dim + self.dim_feature_x, z_dim)
        # generation
        self.theta_norm_1 = NormalParam(hidden_dim, z_dim)
        self.theta_norm_2 = NormalParam(self.dim_feature_z + hidden_dim, n_features)

        self.h0 = nn.Parameter(torch.zeros(hidden_dim))

    def model(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("vrnn", self)
        h = self.h0.expand([b, self.hidden_dim])

        feature_x = self.feature_extra_x(x)
        with pyro.plate("data", b):
            for t in range(1, l + 1):
                z_loc, z_scale = self.theta_norm_1(h)
                with poutine.scale(None, annealing_factor):
                    z_t = pyro.sample(f"z_{t}", dist.Normal(z_loc, z_scale).to_event(1))

                z_t = self.feature_extra_z(z_t)
                z_with_h = torch.cat([z_t, h], dim=1)
                x_loc, x_scale = self.theta_norm_2(z_with_h)
                pyro.sample(f"obs_{t}", dist.Normal(x_loc, x_scale).to_event(1), obs=x[:, t - 1, :])
                x_with_z = torch.cat([feature_x[:, t - 1, :], z_t], dim=1)
                h = self.rnn(x_with_z, h)

    def guide(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("vrnn", self)
        h = self.h0.expand([b, self.hidden_dim])

        feature_x = self.feature_extra_x(x)
        with pyro.plate("data", b):
            for t in range(1, l + 1):
                xt = feature_x[:, t - 1, :]
                h_with_x = torch.cat([h, xt], dim=1)
                z_loc, z_scale = self.phi_norm(h_with_x)
                with poutine.scale(None, annealing_factor):
                    z_t = pyro.sample(f"z_{t}", dist.Normal(z_loc, z_scale).to_event(1))

                z_t = self.feature_extra_z(z_t)
                x_with_z = torch.cat([xt, z_t], dim=1)
                h = self.rnn(x_with_z, h)

    def encode(self, x, h):
        h_with_x = torch.cat([h, x], dim=1)
        z_loc, z_scale = self.phi_norm(h_with_x)
        return z_loc, z_scale

    def decode(self, z, h):
        z_with_h = torch.cat([z, h], dim=1)
        x_loc, x_scale = self.theta_norm_2(z_with_h)
        return x_loc, x_scale

    def forward(self, x, return_prob=False):
        b, l, _ = x.shape
        h = x.new_zeros([b, self.hidden_dim])
        res = torch.zeros_like(x)
        res_scale = torch.zeros_like(x)
        x = self.feature_extra_x(x)
        for t in range(l):
            xt = x[:, t, :]
            z_loc, z_scale = self.encode(xt, h)
            z = dist.Normal(z_loc, z_scale).to_event(1).sample([self.n_sample])
            z = self.feature_extra_z(z.mean(0))
            res[:, t, :], res_scale[:, t, :] = self.decode(z, h)
            x_with_z = torch.cat([xt, z], dim=1)
            h = self.rnn(x_with_z, h)

        return res if not return_prob else (res, res_scale)


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
            z_t = dist.Normal(z_loc, z_scale).to_event(1).sample([self.n_sample])
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


class TransformerVAE(nn.Module):

    def __init__(self, n_features=1, d_model=256, z_dim=4, nhead=8, nlayers=6, dim_feedforward=1024, dropout=0.1,
                 num_nf=10):
        super(TransformerVAE, self).__init__()
        self.z_dim = z_dim

        # inference
        self.phi_pos_encoder = TimePositionalEncoding(d_model, batch_first=True)
        self.phi_x_embedding = Conv1DEmbedding(n_features, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.phi_transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.phi_dense = MLPEmbedding(d_model, d_model, dropout=0.5)
        self.phi_norm = NormalParam(d_model, z_dim)

        self.phi_nf = nn.ModuleList(dist.transforms.Planar(z_dim) for _ in range(num_nf))

        # generation
        self.theta_rnn = nn.GRU(z_dim, d_model, batch_first=True)
        self.theta_dense = MLPEmbedding(d_model, d_model, dropout=0.5)
        self.theta_norm = NormalParam(d_model, n_features)

    def model(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("tfVAE", self)
        matrix = torch.diag(x.new_ones(self.z_dim))
        base_dist = dist.MultivariateNormal(x.new_zeros(self.z_dim), matrix)
        ghmm = dist.GaussianHMM(base_dist, matrix, base_dist, matrix, base_dist, duration=l)

        with pyro.plate("data", b):
            with poutine.scale(None, annealing_factor):
                z = pyro.sample("z", ghmm)
            x_loc, x_scale = self.decode(z)
            pyro.sample("obs", dist.Normal(x_loc, x_scale).to_event(2), obs=x)

    def guide(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("tfVAE", self)
        with pyro.plate("data", b):
            z_loc, z_scale = self.encode(x)
            base_dist = dist.Normal(z_loc, z_scale).to_event(2)
            with poutine.scale(None, annealing_factor):
                pyro.sample("z", dist.TransformedDistribution(base_dist, [nf for nf in self.phi_nf]))

    def encode(self, x):
        embedding = self.phi_x_embedding(x) + self.phi_pos_encoder(x)
        mask = x.new_full((x.size(1), x.size(1)), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        res = self.phi_transformer_encoder(embedding, mask=mask)
        res = self.phi_dense(res)
        z_loc, z_scale = self.phi_norm(res)
        return z_loc, z_scale

    def decode(self, z):
        res, _ = self.theta_rnn(z)
        res = self.theta_dense(res)
        x_loc, x_scale = self.theta_norm(res)
        return x_loc, x_scale

    def forward(self, x, return_prob=True):
        z_loc, z_scale = self.encode(x)
        base_dist = dist.Normal(z_loc, z_scale).to_event(2)
        tf_dist = dist.TransformedDistribution(base_dist, [nf for nf in self.phi_nf])
        z = tf_dist.sample()
        x_loc, x_scale = self.decode(z)
        return x_loc if not return_prob else (x_loc, x_scale)
