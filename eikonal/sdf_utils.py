import skimage
import numpy as np
import torch
from shapely import geometry
import abc

class Shape(metaclass=abc.ABCMeta):
    def __init__(self,n_input):
        self.n_input = n_input

    @abc.abstractmethod
    def get_points(self):
        pass

    @abc.abstractmethod
    def measure_distance(self, points):
        pass

class Sphere(Shape):
    def get_points(self):
        pnts = torch.randn(self.n_input, 3, dtype=torch.float)
        # pnts = torch.empty(self.n_input, 3).uniform_(-1.0, 1.0)
        pnts = pnts / pnts.norm(p=2, dim=-1).unsqueeze(-1)
        return pnts

    def sphere_sdf(self, x):
        return x.norm(p=2, dim=1) - 1.0

    def measure_distance(self, points):
        return self.sphere_sdf(points)

class Plane(Shape):
    def get_points(self):
        theta = np.radians(60)
        c, s = np.cos(theta), np.sin(theta)
        Rx = torch.tensor([(1, 0, 0 ), (0, c, -s), (0, s, c)], dtype=torch.float)
        Ry = torch.tensor([(c, 0, s), (0, 1, 0), (-s, 0, c)], dtype=torch.float)

        pnts = torch.cat([torch.empty(self.n_input, 2, dtype=torch.float).uniform_(-1.0, 1.0),
                       torch.zeros(self.n_input, 1, dtype=torch.float)],
                      dim=1)

        # pnts = torch.mm(torch.mm(pnts, Rx), Ry)
        return pnts

    def measure_distance(self, points):
        return points[:, 2]


class LShape(Shape):
    def get_points(self):
        t = 0.7
        points = np.array([[-t, 0], [0, -t], [t, 0], [0, t], [-t, t], [-t, -t], [t, -t], [0.1 * t, 0.1 * t],
                           [-t, 0.5*t], [-t, -0.5*t], [0.5*t, -t], [-0.5*t, -t], [-0.5*t, t], [t, -0.5*t],
                           [0.05 * t, 0.55 * t], [0.55 * t, 0.05 * t]],
                          dtype='float32')
        return torch.tensor(points,dtype=torch.float)

    def measure_distance(self, points):
        return 0

class Square(Shape):
    def get_points(self):
        t = 0.6
        points = np.array([[-t, t], [-t, -t], [t, -t], [t, t], [-t, 0], [0, -t], [t, 0], [0, t]],
                          dtype='float32')
        return torch.tensor(points,dtype=torch.float)

    def measure_distance(self, points):
        return 0

class Random(Shape):
    def get_points(self):
        torch.manual_seed(4444)
        points = torch.empty(self.n_input, 2, dtype=torch.float).uniform_(-1.0, 1.0)
        torch.initial_seed()
        return points

    def measure_distance(self, points):
        return 0

class Pakmen(Shape):
    def get_points(self):
        points = HalfCircle(8)
        points = points[points[:, 0] >= -0.5]
        return points

    def measure_distance(self, points):
        return 0

class Line(Shape):
    def get_points(self):
        # return torch.cat([torch.tensor(np.linspace(-1.0, 1.0, n),dtype=torch.float).unsqueeze(1),torch.zeros(n,1,dtype=torch.float)],dim=1)
        theta = np.radians(60)
        c, s = np.cos(theta), np.sin(theta)
        R = torch.tensor([(c, -s), (s, c)], dtype=torch.float)

        torch.manual_seed(401)
        a = torch.cat([torch.empty(self.n_input, 1, dtype=torch.float).uniform_(-1.0, 1.0), torch.zeros(self.n_input, 1, dtype=torch.float)],
                      dim=1)

        line_points = torch.mm(a, R)
        torch.initial_seed()

        # normals = torch.tensor([[0, 1]], dtype=torch.float).repeat(self.n_input, 1)
        # normals[0] = torch.tensor([[0.3122, 0.95]], dtype=torch.float)
        # normals[3] = torch.tensor([[-0.5268, 0.85]], dtype=torch.float)
        # normals[5] = torch.tensor([[-0.4359, 0.9]], dtype=torch.float)
        # normals[6] = torch.tensor([[-0.4750, 0.88]], dtype=torch.float)
        # normals[7] = torch.tensor([[0.6258, 0.78]], dtype=torch.float)
        # normals = torch.mm(normals,R)

        return line_points

    def measure_distance(self, points): #TODO
        return (-torch.sqrt(torch.tensor(3,dtype=torch.float)) * points[:,0] - points[:,1]) / 2.0

class DotsNormals(Shape):
    def get_points(self):
        t = 0.5
        points = np.array([[-t, t], [-t, -t], [t, -t], [t, t]],
                          dtype='float32')
        points = torch.tensor(points, dtype=torch.float)
        # n = 0.7071
        # n = -0.7071
        # normals = torch.tensor([[-n, n], [-n, -n], [n, -n], [n, n]]
        #                        , dtype=torch.float)
        # normals = torch.tensor([[0, 1], [0, -1], [0, -1], [0, 1]]
        #                        , dtype=torch.float)

        n = 0.7071
        normals = torch.tensor([[-n, n], [-n, -n], [1, 0], [-n, -n]]
                               , dtype=torch.float)

        return points, normals

    def measure_distance(self, points):
        return 0

class LineCrazy(Shape):
    def get_points(self):
        # return torch.cat([torch.tensor(np.linspace(-1.0, 1.0, n),dtype=torch.float).unsqueeze(1),torch.zeros(n,1,dtype=torch.float)],dim=1)
        theta = np.radians(60)
        c, s = np.cos(theta), np.sin(theta)
        R = torch.tensor([(c, -s), (s, c)], dtype=torch.float)

        torch.manual_seed(401)
        a = torch.cat([torch.empty(self.n_input, 1, dtype=torch.float).uniform_(-1.0, 1.0), torch.zeros(self.n_input, 1, dtype=torch.float)],
                      dim=1)

        line_points = torch.mm(a, R)
        n = 8
        idx = torch.randperm(line_points.shape[0])[:n]
        # rand_points = line_points[idx].clone() + torch.randn(n, 2) * 0.3
        # rand_points[3].add_(torch.tensor([0.35, -0.6], dtype=torch.float))
        # rand_points[1].add_(torch.tensor([-0.05, -0.05], dtype=torch.float))
        # rand_points[2].add_(torch.tensor([0.1, 0.2], dtype=torch.float))
        rand_points = torch.empty(n, 2).uniform_(-0.9, 0.9)
        torch.initial_seed()

        return torch.cat([line_points, rand_points], dim=0)

    def measure_distance(self, points): #TODO
        return (-torch.sqrt(torch.tensor(3,dtype=torch.float)) * points[:,0] - points[:,1]) / 2.0

class HalfCircle(Shape):
    def get_points(self):
        return self.measure_sdf(self.circle, self.n_input)

    def measure_distance(self, points):  # TODO check
        return self.circle(points)

    def circle(self, x):
        return x.norm(p=2, dim=1) - 0.8

    def measure_sdf(self, sdf, n):
        x = np.linspace(-1.0, 1.0, 500)
        y = np.linspace(-1.0, 1.0, 500)
        xx, yy = np.meshgrid(x, y)

        positions = torch.tensor(np.vstack([xx.ravel(), yy.ravel()]).T, dtype=torch.float).cuda()
        z = []
        for i, pnts in enumerate(torch.split(positions, 100000, dim=0)):
            z.append(sdf(pnts).detach().cpu().numpy())
        z = np.concatenate(z, axis=0)

        zero_level_set = skimage.measure.find_contours(z.reshape(x.shape[0], y.shape[0]).T, 0.0)
        zero_level_set = (zero_level_set[0] - np.array([500 // 2, 500 // 2])) / (500 // 2)

        # uni_points = torch.empty(20, 2).uniform_(-0.8, 0.8)

        zero_level_set = zero_level_set[zero_level_set[:,0] >= -0.3]
        torch.manual_seed(4)
        idx = torch.randperm(zero_level_set.shape[0])[:n]
        torch.initial_seed()
        zero_level_set = zero_level_set[idx]

        # return torch.cat([torch.tensor(zero_level_set,dtype=torch.float), uni_points], dim=0)
        return torch.tensor(zero_level_set,dtype=torch.float)



class Snowflake(Shape):
    def __init__(self, n_input):
        super(Snowflake, self).__init__(n_input)
        n = 2
        ir = 1.5
        rep = 3
        self.points = self.sample_snow_flake(n, ir, rep)
        # self.points = self.points[self.points[:,0] >= -0.1]
        self.poly = geometry.Polygon(self.points)

    def get_points(self):
        return torch.tensor(self.points, dtype=torch.float)

    def measure_distance(self, points): #TODO
        return 0

    def getp(self, p1, p2):
        z = complex(p2[0] - p1[0], p2[1] - p1[1])
        zr = complex(1 / 2, -3 ** (1 / 2) / 2)
        z = z * zr
        p = [z.real + p1[0], z.imag + p1[1]]
        return p

    def gps(self, n, ir):
        ip1 = [0, 0]
        if n == 0:
            ip2 = [ir, 0]
            ip3 = [ir / 2, ir / 2 * (3 ** (1 / 2))]
            return [ip1, ip2, ip3]
        else:
            points = self.gps(n - 1, ir)
            points.append(ip1)
            i = 0
            ls = []
            while (1):
                p1 = points[i]
                p2 = points[i + 1]
                p11 = [p1[0] + (p2[0] - p1[0]) / 3, p1[1] + (p2[1] - p1[1]) / 3]
                p12 = [p1[0] + (p2[0] - p1[0]) / 3 * 2, p1[1] + (p2[1] - p1[1]) / 3 * 2]
                p = self.getp(p11, p12)
                ls.append(p1)
                ls.append(p11)
                ls.append(p)
                ls.append(p12)
                i = i + 1
                if len(points) - 1 < i + 1:
                    break
            return ls

    def sample_snow_flake(self, n, ir, rep):
        ps = self.gps(n, ir)
        ps = np.array(ps)
        points = np.array([])

        for p1, p2 in zip(ps, np.roll(ps, 1, axis=0)):
            p_new = np.stack((np.linspace(p1[0], p2[0], rep), np.linspace(p1[1], p2[1], rep)), axis=-1)
            points = np.concatenate((points, p_new[:-1])) if len(points) > 0 else p_new[:-1]
        points -= np.mean(points, axis=0)
        return points
