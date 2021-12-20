import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
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


class VRNN(nn.Module):
    """ Original paper: A Recurrent Latent Variable Model for Sequential Data (https://arxiv.org/abs/1506.02216)
    It's a implementation based on pyro.
    """

    def __init__(self, n_features=1, hidden_dim=256, z_dim=4):
        super(VRNN, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRUCell(input_size=n_features + z_dim, hidden_size=hidden_dim)
        self.phi_norm = NormalParam(hidden_dim + n_features, z_dim)
        self.theta_norm_1 = NormalParam(hidden_dim, z_dim)
        self.theta_norm_2 = NormalParam(z_dim + hidden_dim, n_features)

    def model(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("vrnn", self)
        h = x.new_zeros([b, self.hidden_dim])

        with pyro.plate("data", b):
            for t in range(1, l + 1):
                z_loc, z_scale = self.theta_norm_1(h)
                with poutine.scale(None, annealing_factor):
                    z_t = pyro.sample(f"z_{t}", dist.Normal(z_loc, z_scale).to_event(1))

                z_with_h = torch.cat([z_t, h], dim=1)
                x_loc, x_scale = self.theta_norm_2(z_with_h)
                pyro.sample(f"obs_{t}", dist.Normal(x_loc, x_scale).to_event(1), obs=x[:, t - 1, :])
                x_with_z = torch.cat([x[:, t - 1, :], z_t], dim=1)
                h = self.rnn(x_with_z, h)

    def guide(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        h = x.new_zeros([b, self.hidden_dim])

        with pyro.plate("data", b):
            for t in range(1, l + 1):
                h_with_x = torch.cat([h, x[:, t - 1, :]], dim=1)
                z_loc, z_scale = self.phi_norm(h_with_x)
                with poutine.scale(None, annealing_factor):
                    z_t = pyro.sample(f"z_{t}", dist.Normal(z_loc, z_scale).to_event(1))

                x_with_z = torch.cat([x[:, t - 1, :], z_t], dim=1)
                h = self.rnn(x_with_z, h)

    def encode(self, x, h):
        h_with_x = torch.cat([h, x], dim=1)
        z_loc, z_scale = self.phi_norm(h_with_x)
        z = dist.Normal(z_loc, z_scale).to_event(1).sample()
        # z = reparameterization(z_loc, z_scale)
        x_with_z = torch.cat([x, z], dim=1)
        h = self.rnn(x_with_z, h)
        return z, h

    def decode(self, z, h, return_prob=False):
        z_with_h = torch.cat([z, h], dim=1)
        x_loc, x_scale = self.theta_norm_2(z_with_h)
        return x_loc if not return_prob else (x_loc, x_scale)

    def forward(self, x, return_prob=False):
        b, l, _ = x.shape
        h = x.new_zeros([b, self.hidden_dim])
        res = torch.zeros(x.shape)
        res_scale = torch.zeros(x.shape)
        for t in range(l):
            xt = x[:, t, :]
            z, h_next = self.encode(xt, h)
            if return_prob:
                res[:, t, :], res_scale[:, t, :] = self.decode(z, h, return_prob)
            else:
                res[:, t, :] = self.decode(z, h, return_prob)
            h = h_next

        return res if not return_prob else (res, res_scale)
