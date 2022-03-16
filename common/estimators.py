import torch

def exact_divergence(f, x):
    if not x.requires_grad:
        x.requires_grad = True
    if isinstance(f, torch.Tensor):
        f_x = f
    else:
        f_x = f(x)
    div = torch.zeros(x.shape[0]).to(x)
    for i in range(x.shape[1]):
        H_i = torch.autograd.grad(f_x[:, i].sum(), x, create_graph=True)[0]
        div += H_i[:, i]
    return div
