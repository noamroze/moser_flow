import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import time

from common.architectures import (
    parse_activation,
    build_mlp
)
from common.estimators import (
    exact_divergence
)
from common import priors
from common.surface import ImplicitSphere

def build_architecture(input_dim, config, output_dim):

    last_activation = config.get("last_activation", None)
    if last_activation is not None:
        last_activation = parse_activation(last_activation)

    if config["type"] == 'mlp':
        return build_mlp(
                    input_dim,
                    config["hidden_dims"],
                    output_dim,
                    parse_activation(config["activation"]),
                    last_activation,
                    config.get("add_batchnorm", False)
                )
    else:
        raise NotImplementedError


class Moser(nn.Module):
    def __init__(self, input_dim, config, device, monte_carlo_prior):
        super(Moser, self).__init__()

        self.input_dim = input_dim
        self.config = config

        self.prior = priors.build_prior(input_dim, device, config.get("prior"))
        self.monte_carlo_prior = monte_carlo_prior
        self.divergence = lambda f, x: exact_divergence(f, x)
        # Initialize parameterized neural network
        self.v = self.build_base_pde_solution(output_dim=input_dim)

        self.to(device)
        self.device=device
        self.eps = config.get("eps") if config.get("eps", 0) != 0 else 1e-5

        self.initialize_weights()

    def nu(self, x):
        return torch.exp(self.prior.log_prob(x))

    def signed_mu(self, x):
        return (self.nu(x) - self.divergence(self.u, x)).view(-1, 1)

    def mu_minus(self, x):
        return nn.functional.relu(-self.signed_mu(x) + self.eps)

    def mu_plus(self, x):
        return nn.functional.relu(self.signed_mu(x) - self.eps) + self.eps

    def density(self, x):
        return self.mu_plus(x)

    def build_base_pde_solution(self, output_dim):
        raise NotImplementedError
    
    def positivity_loss(self, lambda_plus, lambda_minus, batch_size):
        prior_samples = self.monte_carlo_prior.sample((batch_size,))
        monte_carlo_density = torch.exp(self.monte_carlo_prior.log_prob(prior_samples))
        # integral_approx =  0.5 * (1 / monte_carlo_density) * torch.abs(self.signed_mu(prior_samples) - self.eps).view(batch_size) + self.eps
        integral_approx =  (1 / monte_carlo_density) * (lambda_plus * self.mu_plus(prior_samples) + lambda_minus * self.mu_minus(prior_samples)).view(batch_size)
        return integral_approx

    def forward(self, x):
        return -torch.log(self.mu_plus(x)[:, 0])
         
    def initialize_weights(self):
        def init_xavier(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.v.apply(init_xavier)
        nn.init.zeros_(self.v[-1].weight)

    def u(self, x):
        raise NotImplementedError

    def ode_func(self, t, x):
        out = (self.u(x)
                / ((1-t) * self.nu(x).view(-1, 1)
                + t * self.mu_plus(x)))
        if torch.isnan(out).any():
            raise ValueError("nans in v_t")
        return out
            
    def transport(self, x, is_reverse=False):
        t = torch.FloatTensor([0, 1]).to(self.device)
        if is_reverse:
            t = torch.flip(t, dims=[0])
        return odeint(lambda t, x: self.ode_func(t, x).detach(), x, t, **self.config['ode_args'])[1]
        
    def sample(self, n_samples):
        random_samples = self.prior.sample((n_samples,))
        random_samples.requires_grad = True
        samples = self.transport(random_samples)
        return samples.detach()

    def direct_log_likelihood(self, x):
        t = torch.FloatTensor([1, 0]).to(self.device)
        def adjoint_func(t, states):
            z = states[0]
            ode_func = lambda z: self.ode_func(t, z)
            return ode_func(z).detach(), self.divergence(ode_func, z).detach()
        y0 = (x, torch.zeros(x.shape[0], device=x.device))
        z, log_det = odeint(adjoint_func, y0, t, **self.config['ode_args'])
        return self.prior.log_prob(z[1]) + log_det[1]
        
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

class TorusMoserFlow(Moser):
    def __init__(self, input_dim, config, device):
        self.n_fourier_features = config.get("n_fourier_features", 1)
        super().__init__(input_dim, config, device, monte_carlo_prior=priors.UniformUnitCube(2, device))
        
    def build_base_pde_solution(self, output_dim):
        return build_architecture(2 * self.n_fourier_features * self.input_dim, self.config["architectures"]["potential"], output_dim)

    def u(self, x):
        feature_vector = [torch.sin(np.pi*(i+1)*x) for i in range(self.n_fourier_features)]
        feature_vector += [torch.cos(np.pi*(i+1)*x) for i in range(self.n_fourier_features)]
        feature_vector = torch.cat(feature_vector, dim=1)
        return self.v(feature_vector)

    def transport(self, x):
        z = super().transport(x)
        return (((z + 1) / 2) % 1) * 2 - 1

class ImplicitMoser(Moser):
    def __init__(self, input_dim, config, device, surface):
        surface = surface.to(device)
        uniform_surface_prior = surface.get_uniform_prior(device)
        super(ImplicitMoser, self).__init__(input_dim, config, device, monte_carlo_prior=uniform_surface_prior)
        self.surface = surface
        self.prior = uniform_surface_prior


    def build_base_pde_solution(self, output_dim):
        return build_architecture(self.input_dim, self.config["architectures"]["potential"], output_dim)

    def u(self, x):
        dim = x.shape[1]
        def P(z, is_detached=False):
            def projection(v):
                normal = self.surface.normal(z)
                if is_detached:
                    normal = normal.detach()
                return v - torch.bmm(v.view(-1, 1, dim), normal.view(-1, dim, 1)).view(-1, 1) * normal / (normal ** 2).sum(dim=1, keepdim=True)
            return projection

        if isinstance(self.surface, ImplicitSphere):
            x = x / torch.norm(x, dim=1, keepdim=True)
            Px = P(x)
            return Px(self.v(x))
            
        Px0 = P(x, is_detached=True)
        x0 = x.detach()
        x_surrogate = x0 + Px0(x - x0)
        Px = P(x_surrogate, is_detached=False)
        return Px(self.v(x_surrogate))



