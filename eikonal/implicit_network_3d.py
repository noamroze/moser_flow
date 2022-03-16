# ---------------------------------------------------------------------------------------------------------------------
# 3d surface reconstruction from point-cloud with EikoNet
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import array as arr
import sys
sys.path.append("./eikonal")
import temp
from plyfile import PlyData
from torch.autograd import grad
import os
from datetime import datetime
import GPUtil
import argparse

# ---------------------------------------------------------------------------------------------------------------------
def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def set_available_gpu():
    deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                        excludeUUID=[])
    gpu = deviceIDs[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

# regularization coefficient
ALPHA = 0.1
# normals regularization coefficient
ALPHA2 = 1.0
# sofplus coefficient
BETA = 100
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
        # self.softplus = nn.Softplus()

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
        y = self.forward(x)
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='debug_3d')
    parser.add_argument('--pc_path', type=str)
    args = parser.parse_args()

    device = 'gpu'
    set_available_gpu()

    # output path+name
    exps_folder_name = 'exps'
    expname = args.name
    expdir = os.path.join(os.path.dirname(__file__), exps_folder_name, expname)
    mkdir_ifnotexists(os.path.join('.', exps_folder_name))
    mkdir_ifnotexists(expdir)
    timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
    mkdir_ifnotexists(os.path.join(expdir, timestamp))
    expdir = os.path.join(expdir, timestamp)

    # model parameters
    d_in = 3
    d_out = 1
    dims = [512, 512, 512] #, 512, 512, 512, 512, 512]
    skips = [] # [4]  # before which layers we do the skip connection
    bias = 1.0
    N = 128 **2  # batch size

    # training parameters
    max_epochs = 100001
    learning_rate = 1.0 * 1e-4
    # learning_rate_decay = 0.95
    decrease_lr_every = 0  # 1000
    decrease_lr_by = 1.0  # 0.5
    sigma_nn = 10 #50

    # output surface every
    output_surface_every = 1000

    # create our MLP model
    model = ImplicitNetwork(0, d_in, d_out, dims, skip_in=skips, geometric_init=True)

    if (device == 'gpu'):
        model = model.cuda()

    # optimize model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # load point cloud
    with open(args.pc_path, "rb") as f:
        plydata = PlyData.read(f)
    vertices = np.concatenate([plydata["vertex"]["x"].reshape(-1, 1), plydata["vertex"]["y"].reshape(-1, 1), plydata["vertex"]["z"].reshape(-1, 1)], axis=1).astype("float64")
    point_set = torch.tensor(vertices, dtype=torch.float32)
    
    # normalize shape to 1.0 radius ball
    point_set = point_set-point_set.mean(dim=0)
    scale = np.max(np.linalg.norm(point_set,np.inf,axis=1))
    point_set = 1.0*point_set/scale
    point_set_np = point_set.cpu().numpy()
    sigma_set = torch.ones(len(point_set)) * 0.01

    for t in range(max_epochs):
        # use per point sigma defined above to sample points
        random_idx = torch.randperm(point_set.shape[0])[:N]
        S = torch.index_select(point_set, 0, random_idx).cuda()
        sigmas = torch.index_select(sigma_set, 0, random_idx).cuda()
        X_normal = (torch.randn(1, S.shape[0], d_in).cuda() * sigmas.unsqueeze(0).unsqueeze(2).repeat(
            1, 1, 3) + S.unsqueeze(0).repeat(1, 1, 1)).reshape(1 * S.shape[0], d_in).cuda()

        X_general = torch.empty(S.shape[0] // 2, d_in).uniform_(-1.0, 1.0).cuda()
        X = torch.cat([X_normal, X_general], 0)

        # compute loss
        Y = model(S)
        grad = model.gradient(X)

        grad_norm = grad[:,0,:].norm(2, dim=1)
        grad_loss = ((grad_norm - 1) ** 2).mean()

        loss_fn = (torch.abs(Y)).mean() + ALPHA * grad_loss #+ ALPHA2 * surface_normals_loss

        # print loss and grad loss every 500 epochs
        if divmod(t, 100)[1] == 0:
            print(expname, timestamp, t, 'loss =', loss_fn.item(), 'grad_loss =', grad_loss.item())

        # backward pass
        optimizer.zero_grad()
        loss_fn.backward()
        optimizer.step()

        # output surface in middle epochs, if required
        if (t > 0) and (output_surface_every > 0) and (np.mod(t, output_surface_every) == 0):
            temp.plot_surface(with_points=True,
                              points=S.detach(),
                              eik_points=X.detach(),
                              decoder=model,
                              latent=None,
                              path=expdir,
                              epoch='iteration__' + str(t),
                              in_epoch=0,
                              shapefile='',
                              resolution=100,
                              mc_value=0,
                              is_uniform_grid=False, verbose=True, save_html=True, save_ply=True)
            temp.plot_cuts(sdf=model,
                           path=expdir,
                           epoch=t,
                           latent=None)
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
    temp.plot_surface(with_points=True,
                      points=S.detach(),
                      eik_points=X.detach(),
                      decoder=model,
                      latent=None,
                      path=expdir,
                      epoch='iteration__' + str(t),
                      in_epoch=0,
                      shapefile='',
                      resolution=500,
                      mc_value=0,
                      is_uniform_grid=False, verbose=True, save_html=True, save_ply=True)

    print('end')
