import torch
from torch import nn
import numpy as np
from .train_misc import *
from common.priors import build_prior
from .lib.layers import diffeq_layers
from .lib.layers.odefunc import NONLINEARITIES

def positional_encoding(x, n_fourier_features):
    feature_vector = [torch.sin(np.pi*(i+1)*x) for i in range(n_fourier_features)]
    feature_vector += [torch.cos(np.pi*(i+1)*x) for i in range(n_fourier_features)]
    return torch.cat(feature_vector, dim=1)

class TorusODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(
        self, hidden_dims, input_shape, layer_type="concat", nonlinearity="softplus", n_fourier_features=1
    ):
        super(TorusODEnet, self).__init__()
        base_layer = {
            "ignore": diffeq_layers.IgnoreLinear,
            "hyper": diffeq_layers.HyperLinear,
            "squash": diffeq_layers.SquashLinear,
            "concat": diffeq_layers.ConcatLinear,
            "concat_v2": diffeq_layers.ConcatLinear_v2,
            "concatsquash": diffeq_layers.ConcatSquashLinear,
            "blend": diffeq_layers.BlendLinear,
            "concatcoord": diffeq_layers.ConcatLinear,
        }[layer_type]

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = 2 * n_fourier_features * input_shape[0]
        for dim_out in hidden_dims + (input_shape[0],):
            layer = base_layer(hidden_shape, dim_out)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])
            hidden_shape = dim_out

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])
        self.n_fourier_features = n_fourier_features

    def forward(self, t, y):
        dx = positional_encoding(y, self.n_fourier_features)
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx

def build_model_torus(args, dims, regularization_fns=None):
    hidden_dims = tuple(map(int, args.dims.split("-")))

    def build_cnf():
        diffeq = TorusODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dims,),
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
            n_fourier_features=args.n_fourier_features
        )
        odefunc = layers.ODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            residual=args.residual,
            rademacher=args.rademacher,
        )
        cnf = layers.CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            regularization_fns=regularization_fns,
            solver=args.solver,
        )
        return cnf

    chain = [build_cnf() for _ in range(args.num_blocks)]
    if args.batch_norm:
        bn_layers = [layers.MovingBatchNorm1d(dims, bn_lag=args.bn_lag) for _ in range(args.num_blocks)]
        bn_chain = [layers.MovingBatchNorm1d(dims, bn_lag=args.bn_lag)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = layers.SequentialFlow(chain)

    set_cnf_options(args, model)

    return model

class FFJORDTorusModel(nn.Module):
    def __init__(self, args, dims, device):
        super(FFJORDTorusModel, self).__init__()
        self.dims = dims
        self.cnf = build_model_torus(args, dims)
        self.prior = build_prior(dims, device, args.prior)
        self.device = device
        self.to(device)

    def forward(self, x):
        model = self.cnf
        zero = torch.zeros(x.shape[0], 1).to(x)

        # transform to z
        z, delta_logp = model(x, zero)
        z = (((z + 1) / 2) % 1) * 2 - 1

        # compute log q(z)
        logpz = self.prior.log_prob(z).view(-1, 1).sum(1, keepdim=True)

        logpx = logpz - delta_logp
        return -logpx

    def sample(self, n_samples):
        samples = self.prior.sample((n_samples,))
        return self.transport(samples)

    def transport(self, x):
        with torch.no_grad():
            z = self.cnf(x, reverse=True)
            z = (((z + 1) / 2) % 1) * 2 - 1
        return z

    def direct_log_likelihood(self, x):
        return -self.forward(x)

    def density(self, x):
        return torch.exp(self.direct_log_likelihood(x))

    def ode_func(self, t, x):
        t = 1 - t
        t = torch.tensor(t).to(x)
        t *= self.cnf.chain[0].sqrt_end_time ** 2
        
        return self.cnf.chain[0].odefunc.diffeq(t, x)