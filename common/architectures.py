import torch.nn as nn

ACTIVATIONS = {
    "tanh": nn.Tanh(),
    "softplus": nn.Softplus(),
    "softplus100": nn.Softplus(100),
    "softplus200": nn.Softplus(200),
    "relu": nn.ReLU(),
}

def parse_activation(activation_name):
    return ACTIVATIONS[activation_name]

def build_mlp(input_dim, 
            hidden_dims,
            output_dim,
            activation,
            last_activation=None,
            add_batchnorm=False):
    layers = [
        nn.Flatten(), 
        nn.Linear(input_dim, hidden_dims[0], bias=True), 
        activation
    ]
    if add_batchnorm:
        layers.append(nn.BatchNorm1d(hidden_dims[0]))

    for i in range(1, len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1], bias=True))
        layers.append(activation)
        if add_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
    layers.append(nn.Linear(hidden_dims[-1], output_dim, bias=True))
    if last_activation is not None:
        layers.append(last_activation)
    return nn.Sequential(*layers)