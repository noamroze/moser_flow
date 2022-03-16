import numpy as np
import torch

def coordinates_to_xyz(theta, phi):
    if isinstance(theta, torch.Tensor):
        lib = torch
        concatenate = torch.cat
    else:
        lib = np
        concatenate = np.concatenate

    x = lib.sin(theta) * lib.cos(phi)
    y = lib.sin(theta) * lib.sin(phi)
    z = lib.cos(theta)
    return concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], 1)

def xyz_to_coordinates(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    if isinstance(points, torch.Tensor):
        acos = torch.acos
        atan2 = torch.atan2
    else:
        acos = np.arccos
        atan2 = np.arctan2
    
    theta = acos(z)
    phi = atan2(y, x)
    return theta, phi