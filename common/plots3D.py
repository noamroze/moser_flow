import matplotlib.pyplot as plt
import numpy as np
import torch
import plotly.graph_objs as go
import trimesh
from evtk import hl, vtk
from common.utils import run_func_in_batches

def colored_mesh_to_vtk(file_path, mesh, face_colors, name):
    n_cells = len(mesh.faces)
    vertices = np.array(mesh.vertices)
    return hl.unstructuredGridToVTK(
        path=file_path,
        x=vertices[:, 0].copy(order='F'),
        y=vertices[:, 1].copy(order='F'),
        z=vertices[:, 2].copy(order='F'),
        connectivity=np.array(mesh.faces.reshape(-1)),
        offsets=np.arange(start=3, stop=3*(n_cells + 1), step=3, dtype='uint32'),
        cell_types = np.ones(n_cells, dtype='uint8')*vtk.VtkTriangle.tid,
        cellData={"name": face_colors.reshape(-1)}
    )

def plot_function_on_surface(name, f, surface, device, n_grid_points, limits=(-1, 1), colorscale=None):
    triangulation = surface.mesh
    points = triangulation.triangles.mean(axis=1)
    verts = triangulation.vertices
    I, J, K = triangulation.faces.transpose()
    face_colors = run_func_in_batches(f, torch.tensor(points.astype("float32")).to(device), 100000, 1).cpu().numpy()
    torch.cuda.empty_cache()

    if colorscale:
        cmin, cmax = colorscale
    else:
        cmin = None
        cmax = None

    traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name=name,
                            intensity=face_colors, intensitymode="cell", colorscale="Viridis", showscale=True, cmin=cmin, cmax=cmax)]

    fig = go.Figure(data=traces)
    fig.add_traces(traces)
    scene_dict = dict(xaxis=dict(range=limits, autorange=False),
                      yaxis=dict(range=limits, autorange=False),
                      zaxis=dict(range=limits, autorange=False),
                      aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=scene_dict, width=1400, height=1400, showlegend=True)
    return fig

def plot_colored_mesh(mesh, colors, intensitymode, colorscale=None, limits=(-1, 1)):
    verts = mesh.vertices
    I, J, K = mesh.faces.transpose()

    if colorscale:
        cmin, cmax = colorscale
    else:
        cmin = None
        cmax = None

    traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K,
                            intensity=colors, intensitymode=intensitymode, colorscale="Viridis", showscale=True, cmin=cmin, cmax=cmax)]

    fig = go.Figure(data=traces)
    scene_dict = dict(xaxis=dict(range=limits, autorange=False),
                      yaxis=dict(range=limits, autorange=False),
                      zaxis=dict(range=limits, autorange=False),
                      aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=scene_dict, width=1400, height=1400, showlegend=True)
    return fig

def plot_pointcloud(samples, fig=None):
    traces = [go.Scatter3d(x=samples[:, 0], y=samples[:, 1], z=samples[:, 2], mode="markers")]
    if fig is None:
        fig = go.Figure()
        scene_dict = dict(xaxis=dict(range=[-3, 3], autorange=False),
                      yaxis=dict(range=[-3, 3], autorange=False),
                      zaxis=dict(range=[-3, 3], autorange=False),
                      aspectratio=dict(x=1, y=1, z=1))
        fig.update_layout(scene=scene_dict, width=1400, height=1400, showlegend=True)
    
    fig.add_traces(traces)
    return fig


def plot_colored_surface(vertices, colors):
    traces = [go.Scatter3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                            marker={"color": colors, "colorscale": "Viridis", "showscale": True}, mode="markers")]

    fig = go.Figure(data=traces)
    fig.add_traces(traces)
    scene_dict = dict(xaxis=dict(range=[-3, 3], autorange=False),
                      yaxis=dict(range=[-3, 3], autorange=False),
                      zaxis=dict(range=[-3, 3], autorange=False),
                      aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=scene_dict, width=1400, height=1400, showlegend=True)
    return fig

def plot_histogram_on_surface(name, samples, surface, n_grid_points, limits=(-1, 1), std=0.05, percentile_for_scale=95, colorscale=None):
    mesh = surface.mesh
    verts = mesh.vertices
    I, J, K = mesh.faces.transpose()

    closest_points, _, closest_faces = trimesh.proximity.closest_point(mesh, samples)
    unique_faces, counts = np.unique(closest_faces, return_counts=True)
    probs = np.zeros(len(mesh.faces))
    probs[unique_faces] = counts / len(samples)
    densities = probs / mesh.area_faces
    densities[np.isnan(densities)] = 0

    if colorscale:
        cmin, cmax = colorscale
    else:
        cmin = -0.1
        cmax = np.percentile(densities, percentile_for_scale)

    traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name=name,
                            opacity=1.0, intensity=densities, intensitymode="cell", colorscale="Viridis", cmin=cmin, cmax=cmax)]
                            
    fig = go.Figure(data=traces)
    scene_dict = dict(xaxis=dict(range=limits, autorange=False),
                      yaxis=dict(range=limits, autorange=False),
                      zaxis=dict(range=limits, autorange=False),
                      aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=scene_dict, width=1400, height=1400, showlegend=True)
    return fig

def plot_3d_quiver(surface, vector_field, device, limits = (-1, 1)):
    mesh = surface.mesh
    verts = mesh.vertices.astype("float32")
    x = torch.tensor(verts).to(device)
    vectors = vector_field(x).cpu().detach().numpy()
    I, J, K = mesh.faces.transpose()
    traces = [
        go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=I, j=J, k=K, name="surface", opacity=1.0),
        go.Cone(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], u=vectors[:, 0], v=vectors[:, 1], w=vectors[:, 2], sizemode="absolute")
    ]
    fig = go.Figure(data=traces)
    scene_dict = dict(xaxis=dict(range=limits, autorange=False),
                      yaxis=dict(range=limits, autorange=False),
                      zaxis=dict(range=limits, autorange=False),
                      aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=scene_dict, width=1400, height=1400, showlegend=True)
    return fig
