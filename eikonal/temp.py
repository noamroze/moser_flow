
from skimage import measure

import numpy as np
import os
import plotly.graph_objs as go
import plotly.offline as offline
import torch
import skimage
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


def get_cuda_ifavailable(torch_obj):
    if (torch.cuda.is_available()):
        return torch_obj.cuda()
    else:
        return torch_obj

def get_grid(points, resolution):
    eps = 0.1
    input_min = torch.min(points, dim=0)[0].squeeze().cpu().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().cpu().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = get_cuda_ifavailable(torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float))
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}

def get_grid_uniform(points, resolution):
    # x = np.linspace(-1.2, 1.2, resolution)
    x = np.linspace(-2.0, 2.0, resolution)

    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points.cuda(),
            "shortest_axis_length": 4.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}

def get_threed_scatter_trace(points, caption=None, colorscale=None, color=None):

    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    trace = go.Scatter3d(
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        mode='markers',
        name='projection',
        marker=dict(
            size=3,
            line=dict(
                width=2,
            ),
            opacity=0.9,
            colorscale=colorscale,
            showscale=True,
            color=color,
        ), text=caption)

    return trace

def get_twod_scatter_trace(points,name = None, caption = None,colorscale = None,color = None):

    assert points.shape[1] == 2, "2d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "2d scatter plot input points are not correctely shaped "

    trace = go.Scatter(
        x=points[:,0].cpu(),
        y=points[:,1].cpu(),
        mode='markers',
        name=name,
        marker=dict(
            size=70,
            line=dict(
                width=0.1,
            ),
            opacity=0.8,
            colorscale=colorscale,
            showscale=False,
            color=color,
        ), text=caption)

    return trace

def plot_contour(points, grad_points, model, path, epoch, resolution, shape, line=False, normals=None):
    model.eval()

    filename = '{0}/iteration_{1}.html'.format(path, epoch)

    # pnts_val = model(points)[0]
    pnts_val = model(points)
    caption = ["model : {0}".format(val.abs().mean().item()) for val in pnts_val.squeeze()]
    trace_pnts = get_twod_scatter_trace(points,caption=caption, name='input', color='white')

    # sdf_eval = shape.measure_distance(grad_points)
    # model_eval = model(grad_points).squeeze()
    # err = ((model_eval.abs() - sdf_eval.abs()).abs() / torch.clamp(sdf_eval.abs(), min=1.0e-5))

    # grad_pnts_val = model.gradient_penalty(grad_points)
    # caption = ["sdf : {0}; model: {1}; err: {2}".format(sdf_val.item(), model_val.item(), er.item()) for sdf_val, model_val, er in zip(sdf_eval, model_eval, err)]
    # trace_grad_points = [get_twod_scatter_trace(grad_points.detach(), caption=caption,name = 'grad_points')]
    trace_grad_points = []

    x = np.linspace(-1.0, 1.0, resolution)
    y = np.linspace(-1.0, 1.0, resolution)
    xx, yy = np.meshgrid(x, y)
    positions = torch.tensor(np.vstack([xx.ravel(), yy.ravel()]).T, dtype=torch.float).cuda()
    z = []
    for i, pnts in enumerate(torch.split(positions, 100000, dim=0)):
        # z.append(model(pnts)[0].detach().cpu().numpy())
        z.append(model(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    trace_zero_level_set = []
    if line:
        zero_level_set = skimage.measure.find_contours(z.reshape(x.shape[0], y.shape[0]).T, 0.0)
        for a in zero_level_set:
            a = (a - np.array([resolution // 2.0, resolution // 2.0])) / (resolution // 2.0)
            trace_zero_level_set.append(go.Scatter(
                x=a[:, 0],
                y=a[:, 1],
                mode='lines',
                line=dict(color='black', width=10)))

    import plotly.express as px
    trace_contour = go.Contour(x=x,
                        y=y,
                        z=z.reshape(x.shape[0], y.shape[0]),
                        colorscale=px.colors.diverging.Tealrose,#colorscale,
                        contours=dict(
                               # start=-0.9, end=0.9,
                               start=-1.2, end=1.2,
                               # start=-0.6, end=0.6,
                               # size=0.15
                               size=0.2

                        ),
                        showscale=False
                        )

    layout = go.Layout(xaxis = dict(range=[-1,1], showgrid=False, showticklabels=False),
                       yaxis = dict(range=[-1,1], showgrid=False, showticklabels=False, scaleratio=1),
                       width=800,
                       height=800,
                       margin=go.layout.Margin(
                                l=0,
                                r=0,
                                b=0,
                                t=0,
                                pad=4
                            ),
                       )

    if normals is not None:
        fig_quiv = ff.create_quiver(points[:, 0].cpu(), points[:, 1].cpu(), normals[:, 0], normals[:, 1],
                               scale=0.35, arrow_scale=0.35, line=dict(color='aliceblue', width=10))
        trace_zero_level_set.append(fig_quiv.data[0])

    fig = go.Figure(data=[trace_contour] + trace_zero_level_set + [trace_pnts] + trace_grad_points, layout=layout)
    fig.update_layout(showlegend=False)
    offline.plot(fig, filename=filename, auto_open=False)
    # fig.write_image('{0}/fig_{1}.pdf'.format(path, epoch))

    model.train()

def plot_surface(with_points, points, eik_points, decoder, latent, path, epoch, in_epoch, shapefile, resolution, mc_value,
                 is_uniform_grid, verbose, save_html, save_ply, save_vtk=False, sdf_val=None):
    if (is_uniform_grid):
        filename = '{0}/{1}_{2}.html'.format(path, epoch, in_epoch)
    else:
        filename = '{0}/{1}_{2}.html'.format(path, epoch, in_epoch)
    vtk_path = '{0}/sdf_grid_{1}'.format(path, resolution)

    pnts_val = decoder(points)
    pnts_grad = decoder.gradient(points)
    points = points.detach()
    pnts_grad = pnts_grad[:, 0, :].norm(2, dim=1)

    # if sdf_val is not None:
    #     caption = ["decoder : {0}, sdf: {1}".format(val.item(), sdf.item()) for val, sdf in zip(pnts_val.squeeze(), sdf_val)]
    # else:
    #     caption = ["decoder : {0}".format(val.item()) for val in pnts_val.squeeze()]
    caption = ["decoder : {0}, grad: {1}".format(val.item(), g.item()) for val, g in
               zip(pnts_val.squeeze(), pnts_grad)]
    trace_pnts = get_threed_scatter_trace(points[:, -3:], caption=caption)

    eik_pnts_val = decoder(eik_points[:, -3:])
    eik_pnts_grad = decoder.gradient(eik_points[:, -3:])
    eik_points = eik_points.detach()
    caption = ["decoder : {0}, grad: {1}".format(val.item(), g.item()) for val, g in
               zip(eik_pnts_val[:,0:1].squeeze(), eik_pnts_grad[:,0,:].norm(2, dim=1))]
    trace_eik_pnts = get_threed_scatter_trace(eik_points[:, -3:], caption=caption)

    surface = get_surface_trace(points, decoder, latent, resolution, mc_value, is_uniform_grid, verbose,
                                save_ply, vtk_path, save_vtk)
    trace_surface = surface["mesh_trace"]
    layout = go.Layout(title=go.layout.Title(text=shapefile), width=1200, height=800,
                       scene=dict(xaxis=dict(range=[-2, 2], autorange=False),
                                  yaxis=dict(range=[-2, 2], autorange=False),
                                  zaxis=dict(range=[-2, 2], autorange=False),
                                  aspectratio=dict(x=1, y=1, z=1)))
    if (with_points):
        fig1 = go.Figure(data=[trace_pnts, trace_eik_pnts] + trace_surface, layout=layout)
    else:
        fig1 = go.Figure(data=trace_surface, layout=layout)

    if (save_html):
        offline.plot(fig1, filename=filename, auto_open=not torch.cuda.is_available())
    if (not surface['mesh_export'] is None):
        surface['mesh_export'].export(filename.split('.ply')[0] + '.ply', 'ply')

    # verts = torch.from_numpy(surface['mesh_export'].vertices).cuda().float()
    # verts_val, verts_grad = decoder(verts, compute_grad=True)
    # print('verts: norm = {0}, sdf = {1}, grad_norm = {2}'.format(verts.norm(2,1).mean().item(),
    #                                                              verts_val[:,0:1].abs().mean().item(),
    #                                                              verts_grad[:,0,:].norm(2, dim=1).mean().item()))
    # print('------------')


def get_surface_trace(points, decoder, latent, resolution, mc_value, is_uniform, verbose, save_ply, vtk_path, save_vtk=False):

    trace = []
    meshexport = None

    if (is_uniform):
        grid = get_grid_uniform(points[:, -3:], resolution)
    else:
        grid = get_grid(points[:, -3:], resolution)

    z = []

    for i, pnts in enumerate(torch.split(grid['grid_points'], 100000, dim=0)):
        if (verbose):
            print ('{0}'.format(i / (grid['grid_points'].shape[0] // 100000) * 100))

        if (not latent is None):
            pnts = torch.cat([latent.expand(pnts.shape[0], -1), pnts], dim=1)
        z.append(decoder(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    if (not (np.min(z) > mc_value or np.max(z) < mc_value)):

        import trimesh
        z = z.astype(np.float64)

        # if save_vtk:
        #     sdf_grid = np.ascontiguousarray(z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
        #                      grid['xyz'][2].shape[0]).transpose([1, 0, 2]))
        #     # sdf_grid = z.reshape(grid['xyz'][0].shape[0], grid['xyz'][1].shape[0],
        #     #                      grid['xyz'][2].shape[0])
        #     from pyevtk.hl import gridToVTK
        #     gridToVTK(vtk_path, grid['xyz'][0], grid['xyz'][1], grid['xyz'][2], pointData={'sdf': sdf_grid})

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=mc_value,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])
        if (save_ply):
            meshexport = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)

        def tri_indices(simplices):
            return ([triplet[c] for triplet in simplices] for c in range(3))

        I, J, K = tri_indices(faces)

        trace.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                               i=I, j=J, k=K, name='',
                               color='orange', opacity=0.5))
    return {"mesh_trace": trace,
            "mesh_export": meshexport}


def plot_cuts(sdf,path,epoch, latent):
    # onedim_cut = np.linspace(-1., 1., 200)
    onedim_cut = np.linspace(-1., 1., 100)
    xx, yy = np.meshgrid(onedim_cut, onedim_cut)
    xx = xx.ravel()
    yy = yy.ravel()
    min_y = torch.tensor(-0.8, dtype=torch.float)
    max_y = torch.tensor(0.8, dtype=torch.float)
    position_cut = np.vstack(([xx, np.zeros(xx.shape[0]), yy]))
    position_cut = [position_cut + np.array([0., i, 0.]).reshape(-1, 1) for i in np.linspace(min_y, max_y, 10)]

    fig = make_subplots(rows=2, cols=5, subplot_titles=['y = {0}'.format(round(p[1, 0], 2)) for p in position_cut])

    for index, pos in enumerate(position_cut):
        field_input = torch.tensor(pos.T, dtype=torch.float).cuda()

        if latent is not None:
            field_input = torch.cat([latent.unsqueeze(0).repeat(field_input.shape[0], 1), field_input], dim=-1)

        z = []
        for i, pnts in enumerate(torch.split(field_input, 1000, dim=-1)):
            z.append(sdf(pnts).detach().cpu().numpy())
        z = np.concatenate(z, axis=0)

        trace = go.Contour(x=onedim_cut,
                            y=onedim_cut,
                            z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
                            name='y = {0}'.format(round(pos[1, 0], 2)),
                            contours=dict(
                                start=-0.3, end=0.8,
                                size=0.05
                            ),
                            showscale=True if index == 4 else False
                            )

        fig.add_trace(trace, row=index // 5 + 1, col=index % 5 + 1)
        fig.update_xaxes(showticklabels=False, row=index // 5 + 1, col=index % 5 + 1)
        fig.update_yaxes(showticklabels=False, row=index // 5 + 1, col=index % 5 + 1)

    fig.update_layout(title='cuts_y', width=1200, height=800, scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                                          yaxis=dict(range=[-1, 1], autorange=False),
                                                          aspectratio=dict(x=1, y=1)))
    filename = '{0}/cuts_{1}'.format(path, epoch)
    offline.plot(fig, filename='{0}.html'.format(filename), auto_open=False)
