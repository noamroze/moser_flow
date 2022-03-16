import torch
from torch import nn
import numpy as np
import trimesh
from skimage import measure
from eikonal.implicit_network_3d import ImplicitNetwork
from common import priors

class ImplicitSurface(object):
    def __init__(self, sdf):
        self.sdf = sdf
        self.device = "cpu"
        self.mesh = None

    def normal(self, x):
        if not x.requires_grad:
            x.requires_grad = True
        return torch.autograd.grad(self.sdf(x).sum(), x, create_graph=True, retain_graph=True)[0]

    def project(self, x, n_iters=1):
        for i in range(n_iters):
            surface_direction = self.normal(x)
            distance = self.sdf(x)
            x = x - distance.view(-1, 1) * surface_direction / torch.norm(surface_direction, dim=1, keepdim=True)
        return x

    def project_detached(self, x, n_iters=1):
        for i in range(n_iters):
            surface_direction = self.normal(x).detach()
            with torch.no_grad():
                distance = self.sdf(x)
                x = x - distance.view(-1, 1) * surface_direction / torch.norm(surface_direction, dim=1, keepdim=True)
        return x
    
    def to(self, x):
        self.device = x
        if isinstance(self.sdf, nn.Module):
            self.sdf.to(x)
        return self

    def calc_mesh(self, n_grid_points=None, limits=None):
        if self.mesh is not None:
            return self.mesh
        if n_grid_points is None or limits is None:
            raise ValueError("No presaved mesh")
        self.mesh = get_triangulation(self, n_grid_points, *[limits] * 3)
        return self.mesh

    def get_uniform_prior(self, device):
        return priors.UniformMesh(self, device)

class ImplicitSphere(ImplicitSurface):
    def __init__(self):
        super(ImplicitSphere, self).__init__(sdf=lambda x: torch.norm(x, dim=1) - 1)

    def get_uniform_prior(self, device):
        return priors.UniformUnitSphere(dim=3, device=device)

def get_surface(name, **kwargs):
    if name == "sphere":
        return ImplicitSphere()
    
    elif name == "elipsoid":
        a = kwargs["a"]
        b = kwargs["b"]
        c = kwargs["c"]
        def elipsoid_implicit(p):
            x = p[:, 0]
            y = p[:, 1]
            z = p[:, 2]
            return torch.sqrt((x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2) - 1
        surface = ImplicitSurface(sdf=elipsoid_implicit)
        surface._mesh = get_triangulation(surface, 100, (-a-0.1, a+0.1), (-b-0.1, b+0.1), (-c-0.1, c+0.1))
        return surface
        
    elif "mesh" in name:
        checkpoint_path = kwargs["checkpoint"]
        sdf = ImplicitNetwork(**kwargs['sdf_params'])
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        sdf.load_state_dict(checkpoint["model"])
        return ImplicitSurface(sdf)
    
    else:
        raise ValueError("No such surface %s" %name)

def get_triangulation(surface, n_points, x_lim, y_lim, z_lim):
    x = np.linspace(*x_lim, n_points)
    y = np.linspace(*y_lim, n_points)
    z = np.linspace(*z_lim, n_points)
    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float32).to(surface.device)
    with torch.no_grad():
        distances = surface.sdf(grid_points)
    distances = distances.detach().cpu().numpy().astype("float32")
    x_spacing = (x_lim[1] - x_lim[0]) / n_points
    y_spacing = (y_lim[1] - y_lim[0]) / n_points
    z_spacing = (z_lim[1] - z_lim[0]) / n_points
    verts, faces, normals, values = measure.marching_cubes(
        volume=distances.reshape(y.shape[0], x.shape[0], z.shape[0]).transpose([1, 0, 2]),
        level=0,
        spacing=(x_spacing, y_spacing, z_spacing)
    )

    verts = verts + np.array([x_lim[0], y_lim[0], z_lim[0]])
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, vertex_colors=values)
