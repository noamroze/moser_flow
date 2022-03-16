import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import trimesh
import igl
import scipy
from PIL import Image
import plotly.offline as offline

from torch.utils.data import Dataset
from data.toy_data import inf_train_gen
from common import plots2D, plots3D
from common.surface import get_surface
from common.sphere_utils import coordinates_to_xyz, xyz_to_coordinates
from common.utils import run_func_in_batches

import importlib
if importlib.util.find_spec("ext_code") is not None:
    from ext_code import earth_plot


class Dataset2D(Dataset):
    def __init__(self):
        super().__init__()
        self.x_lim = (-1, 1)
        self.y_lim = (-1, 1)
        self.bins = 200

    def initial_plots(self, eval_dir, **kwargs):
        data = self[:][0].numpy()

        fig, ax, self._quad_mesh = plots2D.plot_samples(None, None, data, self.x_lim, self.y_lim, bins=self.bins, vmin=-0.1)
        ax.set_title("source distribution")

        fig.savefig(os.path.join(eval_dir, "toy_data_distribution.png")) 

    def evaluate_model(self, model, epoch, eval_dir=None, logger=None, **args):
        x_lim = self.x_lim
        y_lim = self.y_lim

        fig, ax, _ = plots2D.plot_2D_func(None, None, model.density, model.device, *x_lim, *y_lim, N=800, quad_mesh=self._quad_mesh)
        ax.set_title("generated density")
        plots2D.remove_ticks(ax)
        fig.savefig(os.path.join(eval_dir, "generated_density.png")) 

        plt.close('all')

    def test_model(self, model, test_dir, calc_ode_density=False, **args):
        source_samples = model.prior.sample((len(self),))
        generated_samples = run_func_in_batches(model.transport, source_samples, 100000, out_dim=2)
        fig, ax , _ = plots2D.plot_samples(None, None, generated_samples.detach().cpu(), self.x_lim, self.y_lim, bins=self.bins, quad_mesh=self._quad_mesh)
        plots2D.remove_ticks(ax)
        ax.axis('off')
        ax.set_title("generated samples")
        fig.savefig(os.path.join(test_dir, "generated_samples.png"))
        fig.savefig(os.path.join(test_dir, "generated_samples.pdf"))

        if calc_ode_density:
            ode_density = None
            if "saved_ode_density" in args:
                import pickle
                with open(args["saved_ode_density"], "rb") as f:
                    ode_density = pickle.load(f)
            figs, axes, ode_density = plots2D.plot_density_comparison(model, 100, *self.x_lim, *self.y_lim, quad_mesh=self._quad_mesh, percentile=99.9, ode_density=ode_density, vmax=0.4, vmin=0)
            with open(os.path.join(test_dir, "ode_data.pkl"), "wb") as f:
                pickle.dump(p, f)

            if not os.path.exists(os.path.join(test_dir, "densities comparison")):
                os.makedirs(os.path.join(test_dir, "densities comparison"))
            figs[0].savefig(os.path.join(test_dir, "densities comparison", "mu_plus.png"))
            figs[1].savefig(os.path.join(test_dir, "densities comparison", "mu_minus.png"))
            figs[2].savefig(os.path.join(test_dir, "densities comparison", "ode density.png"))
            figs[3].savefig(os.path.join(test_dir, "densities comparison", "diff.png"))


            


def slice_size(index, list_len):
    if isinstance(index, torch.Tensor):
        if index.dtype == torch.bool:
            assert index.shape == (list_len,)
            return index.sum()
        elif not index.is_floating_point():
            return index.shape[0]
        else:
            raise ValueError("bad dtype %s" %index.dtype)
    stop = index.stop if index.stop is not None else list_len
    start = index.start if index.start is not None else 0
    step = index.step if index.step is not None else 1
    size = (stop - start) / step
    return size

class ImageDataset(Dataset2D):
    def __init__(self, img_path, size, x_lim=(-1, 1), y_lim=(-1, 1), device="cpu"):
        super().__init__()
        self.x_lim = x_lim
        self.y_lim = y_lim
        img = np.array(Image.open(img_path).convert('L'))
        h, w = img.shape
        xx = np.linspace(*x_lim, w)
        yy = np.linspace(*y_lim, h)
        xx, yy = np.meshgrid(xx, yy)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)

        self.h = h
        self.w = w
        self.pixels = np.concatenate([xx, yy], 1)
        self.img = img.max() - img
        self.probs = img.reshape(-1) / img.sum()
        self.size = size
        # self.data = self.sample_data(size).to(device)
        self._counter = 0

        self.bins = min(self.h, self.w)
    
    def sample_data(self, batch_size):
        inds = np.random.choice(int(self.probs.shape[0]), int(batch_size), p=self.probs)
        p = self.pixels[inds]
        return torch.FloatTensor(p.astype("float32"))

    def __getitem__(self, index):
        sample = self.sample_data(slice_size(index, self.size))
        # sample = self.data[index] 
        return sample, 0

    def __len__(self):
        return self.size


class ToyDataSet(Dataset2D):
    def __init__(self, name, size, device="cpu", normalize="norm", margin=1., eps=0.):
        super(ToyDataSet, self).__init__()
        self.size = size
        self.device = "cpu"
        self.name = name
        self.data = inf_train_gen(name, batch_size=size).astype("float32")
        self._quad_mesh = None
        self.eps = eps

    def __getitem__(self, index):
        coin  = np.random.random() > self.eps
        if coin:
            return torch.tensor(self.data[index], device=self.device), 0
        else:
            if isinstance(index, int):
                uniform_noise = torch.zeros((2), device=self.device)
                uniform_noise.uniform_(-1, 1)
                return uniform_noise, 0
            else:
                raise NotImplementedError


    def __len__(self):
        return self.size

class Dataset3D(Dataset):
    @staticmethod
    def fig_to_vtk(fig, mesh, out_dir, name):
        plots3D.colored_mesh_to_vtk(
            os.path.join(out_dir, "%s.vtk" %name),
            mesh,
            fig.data[0].intensity,
            name
        )

    def initial_plots(self, eval_dir, model, **kwargs):
        n_points = 100
        fig = plots3D.plot_histogram_on_surface(r"data", self[:len(self)][0].cpu().detach().numpy(), model.surface, n_points)
        offline.plot(fig, filename='{0}/data.html'.format(eval_dir), auto_open=False)
        self.fig_to_vtk(fig, model.surface.mesh, eval_dir, "data_samples")
        
        self.colorscale = (fig.data[0].cmin, fig.data[0].cmax)
        fig = plots3D.plot_histogram_on_surface(r"uniform_sample", model.monte_carlo_prior.sample((len(self), )).cpu().numpy(), model.surface, n_points)
        offline.plot(fig, filename='{0}/uniform_sample.html'.format(eval_dir), auto_open=False)

    def evaluate_model(self, model, epoch, eval_dir=None, logger=None, **args):
        n_points = 100
        def normalize_density(f):
            def wrapper(x):
                return f(x) / model.prior.normalizing_constant
            return wrapper
        fig = plots3D.plot_function_on_surface(r"$\nu-\nabla\cdot v$", normalize_density(model.signed_mu), model.surface, model.device, n_points, colorscale=self.colorscale)
        offline.plot(fig, filename='{0}/density.html'.format(eval_dir), auto_open=False)
        self.fig_to_vtk(fig, model.surface.mesh, eval_dir, "density")

        plt.close('all')

    def test_model(self, model, run_dir, *args, **kwargs):
        n_points = 100
        random_samples = model.prior.sample((len(self),))
        random_samples.requires_grad = True

        generated_samples = run_func_in_batches(model.transport, random_samples, 50000, 3)
        fig = plots3D.plot_histogram_on_surface(r"generated_samples", generated_samples.cpu().detach().numpy(), model.surface, n_points, colorscale=self.colorscale)
        offline.plot(fig, filename='{0}/generated_samples.html'.format(run_dir), auto_open=False)
        self.fig_to_vtk(fig, model.surface.mesh, run_dir, "generated_samples")
        

class EarthData(Dataset3D):
    def __init__(self, name, data, test_data=None):
        self.name = name
        self.data = self.latlon_to_xyz(data)
        if test_data is not None:
            self.test_data = self.latlon_to_xyz(test_data)
        self.surface = get_surface("sphere")
        # self.surface.calc_mesh(100, (-1, 1))

    def latlon_to_xyz(self, data):
        theta = (90 - data[:, 0]) * np.pi / 180
        phi = (data[:, 1]) * np.pi / 180
        return coordinates_to_xyz(theta, phi)

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return len(self.data)

    def initial_plots(self, eval_dir, **kwargs):
        self.colorscale = None

    def evaluate_model(self, model, epoch, eval_dir=None, logger=None, **args):
        if "earth_plot" in globals():
            figs = earth_plot(self.name, model, torch.tensor(self.data, device=model.device), torch.tensor(self.test_data, device=model.device), model.device, 200)
            for i, fig in enumerate(figs):
                fig.savefig(os.path.join(eval_dir, "earth_plot_%s.png" %(i+1)))
                fig.savefig(os.path.join(eval_dir, "earth_plot_%s.pdf"%(i+1)))

    def test_model(self, model, run_dir, *args, **kwargs):
        pass

class FaceColoredMeshDataset(Dataset3D):
    def __init__(self, surface, face_colors, size):
        self.mesh = surface.mesh
        self.face_colors = face_colors
        self.probs = face_colors * self.mesh.area_faces
        self.probs /= self.probs.sum()
        self.surface = surface
        self.size = size
        # self.data = self.sample(size)

    @staticmethod
    def uniform_triangles_sample(triangles):
        tri_origins = triangles[:, 0]
        tri_vectors = triangles[:, 1:].copy()
        tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

        # randomly generate two 0-1 scalar components to multiply edge vectors by
        random_lengths = np.random.random((len(tri_vectors), 2, 1))

        # points will be distributed on a quadrilateral if we use 2 0-1 samples
        # if the two scalar components sum less than 1.0 the point will be
        # inside the triangle, so we find vectors longer than 1.0 and
        # transform them to be inside the triangle
        random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
        random_lengths[random_test] -= 1.0
        random_lengths = np.abs(random_lengths)

        # multiply triangle edge vectors by the random lengths and sum
        sample_vector = (tri_vectors * random_lengths).sum(axis=1)

        # finally, offset by the origin to generate
        # (n,3) points in space on the triangle
        samples = sample_vector + tri_origins
        return samples

    def sample(self, batch_size):
        inds = np.random.choice(int(self.probs.shape[0]), int(batch_size), p=self.probs)
        triangles = self.mesh.triangles[inds]
        samples = self.uniform_triangles_sample(triangles)
        return torch.FloatTensor(samples.astype("float32"))

    def __getitem__(self, index):
        sample = self.sample(slice_size(index, self.size))
        # sample = torch.as_tensor(self.data[index])
        return sample, 0

    def __len__(self):
        return self.size

    def initial_plots(self, eval_dir, **kwargs):
        super().initial_plots(eval_dir, **kwargs)
        colors = self.probs / self.mesh.area_faces
        fig = plots3D.plot_colored_mesh(self.mesh, colors, intensitymode="cell")
        offline.plot(fig, filename='{0}/original_colors.html'.format(eval_dir), auto_open=False)
        self.colorscale = (np.min(colors), np.max(colors))
        self.fig_to_vtk(fig, self.mesh, eval_dir, "original_colors")

class NoisyLoader(torch.utils.data.DataLoader):
    def __init__(self, std, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.std = std
        self.shuffle = kwargs.get("shuffle", True)

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(len(self.dataset))
        else:
            perm = np.arange(len(self.dataset))
        i = 0
        while i < len(self.dataset):
            idx = perm[i : i + self.batch_size]
            sample, label = self.dataset[idx]
            if self.std > 0:
                sample = sample + self.std * torch.randn_like(sample)
            yield sample, label
            i += self.batch_size
       
        
        

class DataHandler:
    def __init__(self, config, eps):
        training_size = config["training_size"]
        validation_size = config["validation_size"]
        test_size = config["test_size"]
        self.training_set = self.create_dataset(config, training_size, eps)
        self.validation_set = self.create_dataset(config, validation_size, eps)
        self.test_set = self.create_dataset(config, test_size, eps)

    def create_dataset(self, config, size, eps):
        raise NotImplementedError

    def get_dataloaders(self, batch_size, eval_batch_size, **kwargs):
        datasets = [self.training_set, self.validation_set, self.test_set]
        batch_sizes = [batch_size, eval_batch_size, eval_batch_size]
        return (torch.utils.data.DataLoader(dataset, batch_size=batch_size, **kwargs) for dataset, batch_size in zip(datasets, batch_sizes))


class ToyDataHandler(DataHandler):
    def create_dataset(self, config, size, eps):
        return ToyDataSet(config["name"], size, eps=eps, **config.get("args", {}))

class ImageDataHandler(DataHandler):
    def __init__(self, config, eps):
        super().__init__(config, eps)
        pixel_width = (self.training_set.x_lim[1] - self.training_set.x_lim[0]) * (1/self.training_set.w)
        pixel_height = (self.training_set.y_lim[1] - self.training_set.y_lim[0]) * (1/self.training_set.h)

        self.std = config.get("std", 0) * (pixel_width + pixel_height) / 2

    def create_dataset(self, config, size, eps):
        return ImageDataset(config["path"], size)

    def get_dataloaders(self, batch_size, eval_batch_size, **kwargs):
        datasets = [self.training_set, self.validation_set, self.test_set]
        batch_sizes = [batch_size, eval_batch_size, eval_batch_size]
        return (NoisyLoader(self.std, dataset, batch_size, **kwargs) for dataset, batch_size in zip(datasets, batch_sizes))

class EarthDataHandler(DataHandler):
    def __init__(self, config, eps):
        csv_path = "./data/earth_data/%s.csv" % config["name"]
        data = pd.read_csv(csv_path, comment="#", header=0).values.astype("float32")

        training_size = config["training_size"]
        validation_size = config["validation_size"]
        test_size = config["test_size"]
        total_config_size = training_size + validation_size + test_size
        training_size = int((training_size / total_config_size) * len(data))
        validation_size = int((validation_size / total_config_size) * len(data))
        test_size = len(data) - validation_size - training_size

        train_data, val_data, test_data = torch.utils.data.random_split(data, [training_size, validation_size, test_size])
        if validation_size == 0:
            val_data = test_data
        self.training_set = EarthData(config["name"], data[train_data.indices], data[test_data.indices]) 
        self.validation_set = EarthData(config["name"], data[val_data.indices])
        self.test_set = EarthData(config["name"], data[test_data.indices])

class SurfaceDataHandler(DataHandler):
    def __init__(self, config, eps):
        self.surface = get_surface(**config)
        mesh = self.surface.calc_mesh(config["mesh_n_grid_points"], limits=(-1, 1))
        if config["color"] == "simple":
            colors = np.ones(len(mesh.faces))
            colors[mesh.triangles.mean(axis=1)[:, 0] < 0] = 2
        elif config["color"] == "curvature":
            points = mesh.triangles.mean(axis=1)
            colors = trimesh.curvature.discrete_mean_curvature_measure(mesh, points, config["radius"]) ** 2
        elif config["color"] == "laplacian_eigen_function":
            laplacian = -igl.cotmatrix(mesh.vertices, mesh.faces)
            massmatrix = igl.massmatrix(mesh.vertices, mesh.faces)
            k_eigen_value = config["k_eigen_value"]
            eigen_function = scipy.sparse.linalg.eigsh(A=laplacian, M=massmatrix, sigma=0., k=k_eigen_value)[1][:, -1]
            colors = np.maximum(eigen_function[mesh.faces].mean(axis=1), 0)
        else:
            raise ValueError("illegal color for face_colored_mesh: %s" % config["color"])
        self.colors = colors

        super().__init__(config, eps)

    def create_dataset(self, config, size, eps):
        return FaceColoredMeshDataset(self.surface, self.colors, size)

    def get_dataloaders(self, batch_size, eval_batch_size, **kwargs):
        datasets = [self.training_set, self.validation_set, self.test_set]
        batch_sizes = [batch_size, eval_batch_size, eval_batch_size]
        return (NoisyLoader(0, dataset, batch_size, **kwargs) for dataset, batch_size in zip(datasets, batch_sizes))


def get_data_handler(config, eps):
    datahandlers = {
        "toy": ToyDataHandler,
        "image": ImageDataHandler,
        "earth": EarthDataHandler,
        "face_colored_mesh": SurfaceDataHandler
    }
    return datahandlers[config["type"]](config, eps)
