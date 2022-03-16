import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import traceback
from torchdiffeq import odeint
from common.estimators import exact_divergence
from common.utils import run_func_in_batches

MAX_BATCH_SIZE = 100000

def create_fig():
    plt.tight_layout()
    fig = plt.figure(figsize=(5, 5), frameon=False)
    fig.patch.set_visible(False)
    ax = fig.add_axes([0, 0, 1, 1])
    remove_ticks(ax)
    ax.axis('off')
    return fig, ax

def plot_vt_divergence(dataset, model, n_samples, n_timestamps):
    samples = dataset[np.random.choice(len(dataset), n_samples)][0]
    timestamps = torch.linspace(1, 0, n_timestamps, device=model.device)
    results = odeint(model.ode_func, torch.Tensor(samples).to(model.device), timestamps, **model.config['ode_args'])
    n_cols=3
    n_rows = int(np.ceil(n_samples/n_cols))
    div_vt = np.empty((n_samples, n_timestamps))
    for i, t in enumerate(timestamps):
        vt = lambda x: model.ode_func(t, x)
        div_vt[:, i] = exact_divergence(vt, results[i], create_graph=False).detach().cpu().numpy()

    results = results.detach().cpu().numpy()
    n_cols = 3
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols)
    for i in range(n_samples):
        ax2 = axes[i // n_cols, i%n_cols]
        ax2.plot(timestamps.cpu().numpy(), div_vt[i], label="start point=(%2.3f, %2.3f)" %(float(samples[i, 0]), float(samples[i, 1])))
        
    return fig, axes

def plot_density_comparison(model, n_pixels, min_x, max_x, min_y, max_y, quad_mesh=None, ode_density=None, **kwargs):
    """plotting 2D density"""
    fig1, fig2, fig3, fig4 = [plt.figure() for _ in range(4)]
    ax1, ax2, ax3, ax4 = [fig.subplots() for fig in [fig1, fig2, fig3, fig4]]
    for ax in (ax1, ax2, ax3, ax4):
        remove_ticks(ax)

    fig1, ax1, qm = plot_2D_func(fig1, ax1, model.mu_plus, model.device, min_x, max_x, min_y, max_y, N=n_pixels, quad_mesh=quad_mesh)
    ax1.set_title("$\mu_+$")

    fig2, ax2, qm2 = plot_2D_func(fig2, ax2, lambda x: model.mu_minus(x) + 1e-8, model.device, min_x, max_x, min_y, max_y, N=n_pixels, vmin=0, vmax=0.1)
    ax2.set_title("$\mu_-$")
    fig2.colorbar(qm2)

    x = torch.linspace(min_x, max_x, n_pixels, dtype=torch.float32)
    y = torch.linspace(min_y, max_y, n_pixels, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x, y)
    data = torch.cat([grid_x.flatten().unsqueeze(1), grid_y.flatten().unsqueeze(1)], 1).to(model.device)
    if ode_density is None:
        try:
            f = lambda x: torch.exp(model.direct_log_likelihood(x)).detach()
            ode_density = run_func_in_batches(f, data, max_batch_size=100000, out_dim=1).cpu().data.numpy()
        except Exception as err:
            traceback.print_exc()
            return (fig1, fig2, fig3, fig4), (ax1, ax2, ax3)
        ode_density = ode_density.reshape(n_pixels, n_pixels)

    quad_mesh = ax3.pcolormesh(grid_x, grid_y, ode_density, norm=qm.norm, shading="auto")
    ax3.set_title("density by ode")
    ax3.invert_yaxis()


    diff = np.abs(ode_density - model.density(data).detach().cpu().numpy().reshape(n_pixels, n_pixels))
    qm = ax4.pcolormesh(grid_x, grid_y, diff, shading="auto", vmin=0, vmax=0.1)
    ax4.set_title("densities difference")
    ax4.invert_yaxis()

    print(torch.max(model.density(data)))

    fig4.colorbar(qm)

    return (fig1, fig2, fig3, fig4), (ax1, ax2, ax3, ax4), p

def plot_moser_vector_field(model, min_x, max_x, min_y, max_y, N=50, ax=None):
    grid_x = np.linspace(min_x, max_x, N)
    grid_y = np.linspace(min_y, max_y, N)
    x = np.empty((N**2, 2))
    for i in range(N):
        for j in range(N):
            x[i * N + j, 0] = grid_x[i]
            x[i * N + j, 1] = grid_y[j]
    if ax is None:
        fig, ax = plt.subplots()
        color='b'
    else:
        fig = None
        color = 'y'
    f_x = model.u(torch.Tensor(x).to(model.device)).cpu().detach().numpy()
    ax.quiver(x[:, 0], x[:, 1], f_x[:, 0], f_x[:, 1], color=color)
    ax.set_title("grad(a) function")
    ax.set(xlim=(min_x, max_x), ylim=(min_y, max_y))
    return fig, ax

def plot_ode_vector_field(model, min_x, max_x, min_y, max_y, N=50, k=5):
    fig = plt.figure()
    n_rows = 2
    n_cols = int(np.ceil(k/n_rows))
    x = np.empty((N**2, 2))
    grid_x = np.linspace(min_x, max_x, N)
    grid_y = np.linspace(min_y, max_y, N)
    for i in range(N):
        for j in range(N):
            x[i * N + j, 0] = grid_x[i]
            x[i * N + j, 1] = grid_y[j]
    for i in range(k):
        t = i / (k - 1)
        f_x = model.ode_func(t, torch.Tensor(x).to(model.device)).cpu().detach().numpy()
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.quiver(x[:, 0], x[:, 1], f_x[:, 0], f_x[:, 1])
        ax.set_title("time %s" %t)
        ax.set(xlim=(min_x, max_x), ylim=(min_y, max_y))
        ax.invert_yaxis()

    fig.suptitle("ode function over different times")
    return fig, ax

def plot_2D_func(fig, ax, func, device, min_x, max_x, min_y, max_y, quad_mesh=None, N=100, plot_type="pcolormesh", calc_integral=False, logscale=False, **kwargs):
    if fig is None:
        assert ax is None
        fig, ax = plt.subplots()
        ax.axis('equal')
    x = np.linspace(min_x, max_x, N, dtype=np.float32)
    y = np.linspace(min_y, max_y, N, dtype=np.float32)
    x, y = np.meshgrid(x, y)
    x1 = x.flatten()
    y1 = y.flatten()
    points = torch.FloatTensor(np.concatenate([x1[None].T, y1[None].T], axis=1).astype("float32")).to(device)
    z = run_func_in_batches(func, points, max_batch_size=100000, out_dim=1)[:, 0].detach().cpu().numpy()
    z = z.reshape(N, N)
    torch.cuda.empty_cache()
    if plot_type == "pcolormesh":
        if logscale:
            norm = colors.LogNorm(vmin=1e-8, vmax=1e-1)
        else:
            norm = quad_mesh.norm if quad_mesh is not None else None
        out = ax.pcolormesh(x, y, z, norm=norm, shading="auto", **kwargs)
        ax.set(xlim=(min_x, max_x), ylim=(min_y, max_y))
        if calc_integral:
            integral=np.sum(z) * (2./N) ** 2
            ax.text(min_x, max_y + 0.3, "integral=%2.3f" %integral, bbox=dict(facecolor='red', alpha=0.5))
        ax.invert_yaxis()
    
    elif plot_type == "image":
        out = ax.imshow(z, **kwargs)
        # ax.invert_xaxis()
        ax.axis('off')
        plt.tight_layout()
    
    else:
        raise NotImplementedError

    return fig, ax, out

def plot_samples(fig, ax, samples, x_lim, y_lim, **kwargs):
    if fig is None:
        assert ax is None
        fig, ax = plt.subplots()
        ax.axis('equal')
    if isinstance(samples, torch.Tensor):
        samples = samples.numpy()
    if "quad_mesh" in kwargs:
        quad_mesh = kwargs.pop("quad_mesh")
        norm = quad_mesh.norm
    else:
        norm = None
    _, _, _, args = ax.hist2d(samples[:, 0], samples[:, 1], range=[x_lim, y_lim], density=True, norm=norm, **kwargs)
    remove_ticks(ax)
    ax.set(xlim=x_lim, ylim=y_lim)
    ax.invert_yaxis()
    return fig, ax, args

def remove_ticks(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.axis('equal')
    ax.axis('off')
