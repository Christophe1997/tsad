import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from torch import nn

from tsad.config import register
from tsad.models import dvae


@register("vrnn_pyro", "n_features", "hidden_dim", "z_dim", "dropout", "phi_dense", "theta_dense")
class VRNNPyro(dvae.VRNN):

    def __init__(self, *args, **kwargs):
        super(VRNNPyro, self).__init__(*args, **kwargs)

    def model(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("vrnn", self)

        h = x.new_zeros([b, self.hidden_dim])

        with pyro.plate("data", b):
            for t in range(1, l + 1):
                z_loc, z_scale = self.theta_p_z(h)
                with poutine.scale(None, annealing_factor):
                    zt = pyro.sample(f"z_{t}", dist.Normal(z_loc, z_scale).to_event(1))

                x_loc, x_scale, x_dist = self.generate_x(zt, h)
                pyro.sample(f"obs_{t}", x_dist, obs=x[:, t - 1, :])
                h = self.recurrence(x[:, t - 1, :], zt, h)

    def guide(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("vrnn", self)

        h = x.new_zeros([b, self.hidden_dim])

        with pyro.plate("data", b):
            for t in range(1, l + 1):
                xt = x[:, t - 1, :]
                z_loc, z_scale, zdist = self.inference(xt, h)
                with poutine.scale(None, annealing_factor):
                    zt = pyro.sample(f"z_{t}", zdist)

                h = self.recurrence(xt, zt, h)


@register("omni_pyro", "n_features", "hidden_dim", "z_dim", "dropout", "phi_dense", "theta_dense")
class OmniPyro(dvae.Omni):

    def __init__(self, *args, **kwargs):
        super(OmniPyro, self).__init__(*args, **kwargs)
        self.phi_nf = dist.transforms.spline(self.z_dim)

    def model(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("omni", self)

        h = x.new_zeros([b, self.hidden_dim])
        zt_loc, zt_scale = x.new_zeros([b, self.z_dim]), x.new_ones([b, self.z_dim])

        with pyro.plate("data", b):
            for t in range(l + 1):
                with poutine.scale(None, annealing_factor):
                    zt = pyro.sample(f"z_{t}", dist.Normal(zt_loc, zt_scale).to_event(1))
                h = self.theta_rnn(zt, h)
                x_loc, x_scale, x_dist = self.generate_x(h)
                pyro.sample(f"obs_{t}", x_dist, obs=x[:, t - 1, :])
                zt_loc = zt

    def guide(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("omni", self)

        h = x.new_zeros([b, self.hidden_dim])
        zt = x.new_zeros([b, self.z_dim])

        with pyro.plate("data", b):
            for t in range(l + 1):
                xt = x[:, t - 1, :]
                h = self.phi_rnn(xt, h)
                zt_loc, zt_scale, zt_dist = self.inference(h, zt)
                zt_dist_tf = dist.TransformedDistribution(zt_dist, [self.phi_nf])
                with poutine.scale(None, annealing_factor):
                    zt = pyro.sample(f"z_{t}", zt_dist_tf)


@register("rvae_pyro", "n_features", "hidden_dim", "z_dim", "dropout", "phi_dense", "theta_dense")
class RVAEPyro(dvae.RVAE):

    def model(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("rvae", self)

        prior_z = dist.Normal(x.new_zeros([b, self.z_dim]), x.new_ones([b, self.z_dim])).to_event(1)
        h = x.new_zeros([b, self.hidden_dim])

        with pyro.plate("data", b):
            for t in range(1, l + 1):
                with poutine.scale(None, annealing_factor):
                    zt = pyro.sample(f"z_{t}", prior_z)
                h = self.theta_rnn(zt, h)
                x_loc, x_scale, x_dist = self.generate_x(h)
                pyro.sample(f"obs_{t}", x_dist, obs=x[:, t - 1, :])

    def guide(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("rvae", self)

        x_reverse = torch.flip(x, [1])
        xh, _ = self.phi_rnn_1(x_reverse)
        xh_r = torch.flip(xh, [1])
        zh = x.new_zeros([b, self.hidden_dim])
        zt = x.new_zeros([b, self.z_dim])

        with pyro.plate("data", b):
            for t in range(1, l + 1):
                zh = self.phi_rnn_2(zt, zh)
                zt_loc, zt_scale, zt_dist = self.inference(zh, xh_r[:, t - 1, :])
                with poutine.scale(None, annealing_factor):
                    zt = pyro.sample(f"z_{t}", zt_dist)


@register("srnn_pyro", "n_features", "hidden_dim", "z_dim", "dropout", "phi_dense", "theta_dense")
class SRNNPyro(dvae.SRNN):

    def __init__(self, *args, **kwargs):
        super(SRNNPyro, self).__init__(*args, **kwargs)

    def model(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("srnn", self)

        h, _ = self.phi_rnn_1(dvae.lag(x))
        zt = x.new_zeros([b, self.z_dim])

        with pyro.plate("data", b):
            for t in range(1, l + 1):
                ht = h[:, t - 1, :]
                zt_loc, zt_scale, zt_dist = self.generate_z(ht, zt)
                with poutine.scale(None, annealing_factor):
                    zt = pyro.sample(f"z_{t}", zt_dist)
                x_loc, x_scale, x_dist = self.generate_x(zt, ht)
                pyro.sample(f"obs_{t}", x_dist, obs=x[:, t - 1, :])

    def guide(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("srnn", self)

        h, _ = self.phi_rnn_1(dvae.lag(x))
        x_with_h = torch.cat([x, h], dim=-1)
        h, _ = self.phi_rnn_2(torch.flip(x_with_h, [1]))
        h = torch.flip(h, [1])
        zt = x.new_zeros([b, self.z_dim])

        with pyro.plate("data", b):
            for t in range(1, l + 1):
                zt_with_h = torch.cat([zt, h[:, t - 1, :]], dim=-1)
                zt_with_h = self.phi_dense(zt_with_h)
                zt_loc, zt_scale, zt_dist = self.phi_p_z_x(zt_with_h, return_dist=True)
                with poutine.scale(None, annealing_factor):
                    zt = pyro.sample(f"z_{t}", zt_dist)


@register("tfvae_pyro", "n_features", "z_dim", "nhead", "nlayers", "phi_dense", "theta_dense", "dropout",
          d_model="hidden_dim",
          dim_feedforward="dense_dim")
class TransformerVAEPyro(dvae.TransformerVAE):

    def __init__(self, *args, **kwargs):
        super(TransformerVAEPyro, self).__init__(*args, **kwargs)

    def model(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("tfvae", self)

        h = self.encode(dvae.lag(x))
        zt = x.new_zeros([b, self.z_dim])

        with pyro.plate("data", b):
            for t in range(1, l + 1):
                ht = h[:, t - 1, :]
                zt_loc, zt_scale, zt_dist = self.generate_z(ht, zt)
                with poutine.scale(None, annealing_factor):
                    zt = pyro.sample(f"z_{t}", zt_dist)
                x_loc, x_scale, x_dist = self.generate_x(zt, ht)
                pyro.sample(f"obs_{t}", x_dist, obs=x[:, t - 1, :])

    def guide(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("tfvae", self)

        h = self.encode(x, mask_up=False)
        zt = x.new_zeros([b, self.z_dim])

        with pyro.plate("data", b):
            for t in range(1, l + 1):
                zt_with_h = torch.cat([zt, h[:, t - 1, :]], dim=-1)
                zt_with_h = self.phi_dense(zt_with_h)
                zt_loc, zt_scale, zt_dist = self.phi_p_z_x(zt_with_h, return_dist=True)
                with poutine.scale(None, annealing_factor):
                    zt = pyro.sample(f"z_{t}", zt_dist)


@register("ntfvae_pyro", "n_features", "z_dim", "nhead", "nlayers", "dropout", "phi_mask_up", "theta_dense",
          d_model="hidden_dim",
          dim_feedforward="dense_dim")
class NaiveTransformerVAEPyro(dvae.NaiveTransformerVAE):

    def __init__(self, *args, **kwargs):
        super(NaiveTransformerVAEPyro, self).__init__(*args, **kwargs)

    def model(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("ntfvae", self)
        h = self.theta_encode(dvae.lag(x))

        with pyro.plate_stack("data", [b, l]):
            _, _, z_dist_prior = self.generate_z(h)
            with poutine.scale(None, annealing_factor):
                z = pyro.sample("z", z_dist_prior)
            x_loc, x_scale, x_dist = self.generate_x(z, h)
            pyro.sample("obs", x_dist, obs=x)

    def guide(self, x, annealing_factor=1.0):
        b, l, _ = x.shape
        pyro.module("ntfvae", self)

        with pyro.plate_stack("data", [b, l]):
            z_loc, z_scale, z_dist = self.inference(x)
            with poutine.scale(None, annealing_factor):
                z = pyro.sample("z", z_dist)
