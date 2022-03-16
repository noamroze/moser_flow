# ---------------------------------------------------------------------------------------------------------------------
# 2d surface reconstruction from point-cloud with EikoNet
#
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# imports
import torch
import torch.nn as nn
import numpy as np
import temp
import os
from datetime import datetime
import GPUtil
import sdf_utils

# import old.grad_layers as nng

# ---------------------------------------------------------------------------------------------------------------------
def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                    excludeUUID=[])
gpu = deviceIDs[0]
os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

# regularization coefficient
ALPHA = 0.1
# sofplus coefficient
BETA = 100

n_input = 8

# ---------------------------------------------------------------------------------------------------------------------

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            latent_size,
            d_in,
            d_out,
            dims,
            skip_in=(),
            weight_norm=False,
            geometric_init=False,
            bias=1.0,
    ):
        super().__init__()

        dims = [d_in + latent_size] + dims + [d_out]

        self.pc_dim = d_in #
        self.d_in = d_in + latent_size
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            # lin = nng.LinearGrad(dims[l], out_dim)
            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        # self.softplus = nng.SoftplusGrad(beta=100)
        self.softplus = nn.Softplus(beta=100)

    # def forward(self, input, compute_grad=False):
    def forward(self, input):
        '''
        :param input: [shape: (N x d_in)]
        :param compute_grad: True for computing the input gradient. default=False
        :return: x: [shape: (N x d_out)]
                 x_grad: input gradient if compute_grad=True [shape: (N x d_in x d_out)]
                         None if compute_grad=False
        '''
        x = input
        # x_grad = None

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
                # if compute_grad:
                #     skip_grad = torch.eye(self.d_in, device=x.device)[:, -self.pc_dim:].repeat(input.shape[0], 1, 1)#
                #     x_grad = torch.cat([x_grad, skip_grad], 1) / np.sqrt(2)

            # x, x_grad = lin(x, x_grad, compute_grad, l == 0, self.pc_dim)
            x = lin(x)

            if l < self.num_layers - 2:
                # x, x_grad = self.softplus(x, x_grad, compute_grad)
                x = self.softplus(x)

        # return x, x_grad
        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

# --------------------------------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':

    device = 'gpu'

    # output path+name
    exps_folder_name = 'exps'
    expname = 'debug'
    # expdir = os.path.join(os.environ['HOME'], 'data/Projects/Eikonal-Network/{0}/{1}'.format(exps_folder_name, expname))

    expdir = os.path.join('.', exps_folder_name, expname)
    # mkdir_ifnotexists(os.path.join(os.environ['HOME'],
    #                                      'data/Projects/Eikonal-Network/{0}'.format(exps_folder_name)))
    mkdir_ifnotexists(os.path.join('.', exps_folder_name))
    mkdir_ifnotexists(expdir)
    timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
    mkdir_ifnotexists(os.path.join(expdir, timestamp))
    expdir = os.path.join(expdir, timestamp)

    # model parameters
    d_in = 2
    d_out = 1
    dims = [512, 512, 512, 512, 512, 512, 512, 512]
    skips = [4]  # before which layers we do the skip connection
    bias = 1.0
    N = 1  # batch size

    # training parameters
    max_epochs = 100001
    learning_rate = 1.0 * 1e-4
    # learning_rate_decay = 0.95
    decrease_lr_every = 0  # 1000
    decrease_lr_by = 1.0  # 0.5
    sigma_nn = 1 #50

    # output surface every
    output_surface_every = 1000

    # create our MLP model
    model = ImplicitNetwork(0,d_in, d_out, dims, skip_in=skips)

    if (device == 'gpu'):
        model = model.cuda()

    # optimize model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 2D expirements
    shape = sdf_utils.Line(n_input)
    # shape = sdf_utils.LineCrazy(n_input)
    # shape = sdf_utils.HalfCircle(n_input)
    # shape = sdf_utils.Snowflake(n_input)
    # shape = sdf_utils.Square(n_input)
    # shape = sdf_utils.LShape(n_input)
    # shape = sdf_utils.Random(n_input)

    S = shape.get_points()
    print(S.shape)

    # move to GPU
    if device == 'gpu':
        S = S.cuda()

    # compute sigma per point
    n = S.shape[0]
    S1 = S.unsqueeze(0).repeat(n, 1, 1)
    S2 = S.unsqueeze(1).repeat(1, n, 1)
    D = torch.norm(S1 - S2, p=2, dim=2)
    sorted, indices = torch.sort(D, dim=1)
    sigma_max = D.max()
    sigmas = sorted[:, sigma_nn]
    sigmas = sigmas.cuda()

    for t in range(max_epochs):

        X_1 = ((torch.randn(N, S.shape[0], d_in).cuda() * (sigmas.unsqueeze(0).unsqueeze(2).repeat(1, 1, 2)) +
                S.unsqueeze(0).repeat(N, 1, 1)).reshape(N * S.shape[0], d_in)).cuda()
        X_general = torch.empty(S.shape[0] // 2, d_in).uniform_(-1.0, 1.0).cuda()
        X = torch.cat([X_1, X_general], 0)

        # compute loss

        # Y, grad = model(torch.cat([S, X], 0), compute_grad=True)
        # Y = Y[:S.shape[0], 0:1]

        Y = model(S)
        grad = model.gradient(torch.cat([X,S.clone()],dim=0))

        grad_norm = grad[:,0,:].norm(2, dim=1)
        grad_loss = ((grad_norm - 1) ** 2).mean()

        loss_fn = (torch.abs(Y)).mean() + ALPHA * grad_loss

        # print loss and grad loss every 500 epochs
        if divmod(t, 100)[1] == 0:
            print(expname, timestamp, t, 'loss =', loss_fn.item(), 'grad_loss =', grad_loss.item())

        # backward pass
        optimizer.zero_grad()
        loss_fn.backward()
        optimizer.step()

        # output surface in middle epochs, if required
        if (t >= 0) and (output_surface_every > 0) and (np.mod(t, output_surface_every) == 0):
            temp.plot_contour(points=S,
                              grad_points=X,
                              model=model,
                              path=expdir,
                              epoch=t,
                              resolution=500,
                              shape=shape,
                              line=True)
            torch.save(
                {"model": model.state_dict()},
                os.path.join(expdir + "/network_{0}.pth".format(t)))

        # update learning rate, if required
        if (decrease_lr_every > 0) and (np.mod(t, decrease_lr_every) == 0) and (t > 1):
            learning_rate = learning_rate * decrease_lr_by
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    torch.save(
        {"model": model.state_dict()},
        os.path.join(expdir + "/network_{0}.pth".format(t)))

    # plot the zero level set surface
    temp.plot_contour(points=S,
                      grad_points=X,
                      model=model,
                      path=expdir,
                      epoch=t,
                      resolution=1000,
                      shape=shape,
                      line=True)

    print('end')
