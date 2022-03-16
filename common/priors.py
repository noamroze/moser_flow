from torch import distributions
import torch
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

class StandartGaussian(distributions.MultivariateNormal):
    def __init__(self, dim, device, std):
        super(StandartGaussian, self).__init__(torch.zeros(dim, device=device), std * torch.eye(dim, device=device))


class UniformUnitBall(distributions.Distribution):
    def __init__(self, dim, device):
        super(UniformUnitBall, self).__init__()
        self.dim = dim
        self.device = device

    def log_prob(self, value):
        return np.log(1./np.pi) * torch.ones(value.shape[0], device=self.device)

    def sample(self, shape):
        n, = shape
        r = torch.empty((n, 1), device=self.device)
        r.uniform_(0, 1)
        sample = torch.randn(n, self.dim, device=self.device)
        sample /= torch.norm(sample, dim=1, keepdim=True)
        return r**(1./self.dim) * sample

class UniformUnitSphere(distributions.Distribution):
    def __init__(self, dim, device):
        super(UniformUnitSphere, self).__init__()
        self.dim = dim
        self.device = device

    @property
    def normalizing_constant(self):
        return 1

    def log_prob(self, x):
        return np.log(1 / (4 * np.pi)) * torch.ones(x.shape[0]).to(x)

    def sample(self, shape):
        n, = shape
        sample = torch.randn(n, self.dim, device=self.device)
        sample /= torch.norm(sample, dim=1, keepdim=True)
        return sample


class UniformUnitCube(distributions.Distribution):
    def __init__(self, dim, device):
        self.dim = dim
        self.device = device

    def log_prob(self, x):
        return self.dim * np.log(1./2) * torch.ones(x.shape[0], device=self.device)

    def sample(self, shape):
        n, = shape
        out = torch.zeros(n, self.dim, device=self.device)
        out.uniform_(-1, 1)
        return out

class UniformMesh(distributions.Distribution):
    def __init__(self, surface, device):
        self.surface = surface
        self.mesh = surface.mesh
        self.device = device

    @property
    def normalizing_constant(self):
        return self.mesh.area

    def log_prob(self, x):
        return torch.zeros(x.shape[0]).to(x)

    def sample(self, shape):
        return torch.as_tensor(self.mesh.sample(shape).astype("float32"), device=self.device)
        

def build_prior(input_dim, device, prior_name, **kwargs):
    if not prior_name:
        return
    if prior_name == "gaussian":
        return StandartGaussian(input_dim, device, std=kwargs.get("std", 1))
    if prior_name == "uniform_ball":
        return UniformUnitBall(input_dim, device)
    if prior_name == "uniform_cube":
        return UniformUnitCube(input_dim, device)
    else:
        raise ValueError("Illegal prior: %s" %prior_name)