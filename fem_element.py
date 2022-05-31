import numpy as np
import math


class FEMElement:
    def __init__(self, vertices, vertex_ids):
        self.reference_vertices = vertices
        self.deformed_vertices = vertices
        self.velocities = np.zeros((4, 3))
        self.contracted_center = None
        self.vertex_ids = vertex_ids
        self.center = self.calc_center()
        self.contracted = False
        self.contracted_timer = 0
        self.forces = np.zeros((4, 3))
        self.activation_time = 0
        surface_triangle, surface_area,  = self.find_surface_triangle()
        self.surface_area = surface_area
        self.surface_triangle = surface_triangle
        self.dldt = np.zeros(6)
        self.stiffened = False
        self.stiffen_time = 0
        self.is_boundary = self.check_is_boundary()

    def calc_center(self):
        v1, v2, v3, v4 = self.reference_vertices
        return np.divide(v1 + v2 + v3 + v4, 4.0)

    def find_surface_triangle(self):
        r0, r1, r2, r3 = self.reference_vertices
        v0, v1, v2, v3 = self.vertex_ids
        triangles = [[r0, r1, r2], [r0, r1, r3], [r0, r2, r3], [r1, r2, r3]]
        idxs = [[v0, v1, v2], [v0, v1, v3], [v0, v2, v3], [v1, v2, v3]]
        min_theta = math.pi
        surface_triangle = None
        surface_area = 0
        for i, (a, b, c) in enumerate(triangles):
            ab = b - a
            ac = c - a
            normal = np.cross(ab, ac)
            theta = np.arccos(np.dot(normal, self.center) / (np.linalg.norm(normal) * np.linalg.norm(self.center)))
            theta = min(theta, math.pi - theta)
            if theta < min_theta:
                min_theta = theta
                surface_triangle = idxs[i]
                surface_area = np.linalg.norm(ab) * np.linalg.norm(ac) * math.cos(theta) / 2.0
        return surface_triangle, surface_area

    def check_is_boundary(self):
        for v in self.reference_vertices:
            if v[1] < .01:
                return True
        return False

    def update(self, t):
        if self.contracted and t - self.activation_time > 2:
            self.relax()
        if self.stiffened and t - self.stiffen_time > .8:
            self.unstiffen()
        self.compute_forces(t)

    def contract(self, t):
        self.contracted = True
        self.activation_time = t

    def relax(self):
        self.contracted = False

    def stiffen(self, t):
        if not self.contracted:
            self.stiffened = True
            self.stiffen_time = t

    def unstiffen(self):
        self.stiffened = False

    def compute_forces(self, t):
        t *= 150
        self.forces.fill(0)

        if self.contracted:
            time_elapsed = t - self.activation_time * 150
            a = self.activation(time_elapsed)
            jelly_center = np.array([0, self.center[1], 0])
            resting_muscle_fiber = jelly_center - self.center
            F0 = .00003
            LF = np.linalg.norm(jelly_center - self.center)
            L0 = np.linalg.norm(self.center)
            S = 0.4
            F = F0 * math.e**(-((LF/L0 - 1) / S)**2)
            f = F * a

            muscle_dir = resting_muscle_fiber / LF

            for i in range(4):
                self.forces[i] = f * muscle_dir
                continue
        if self.is_boundary:
            ks = .001
        else:
            ks = .003

        spring_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        d = np.zeros((2, 3, 6))
        r = np.zeros((2, 3, 6))
        for i, (i1, i2) in enumerate(spring_pairs):
            d[0, :, i] = self.deformed_vertices[i1]
            d[1, :, i] = self.deformed_vertices[i2]
            r[0, :, i] = self.reference_vertices[i1]
            r[1, :, i] = self.reference_vertices[i2]

        d1d2 = d[0] - d[1]
        RL = np.linalg.norm(r[0] - r[1], axis=0)
        bs = .00025
        dldt = 0
        f = ks * (1 - (RL / np.linalg.norm(d1d2, axis=0))) * d1d2 + bs * dldt

        for i, (i1, i2) in enumerate(spring_pairs):
            self.forces[i1] -= f[:, i]
            self.forces[i2] += f[:, i]

        dir = np.subtract(self.reference_vertices, self.deformed_vertices)
        if self.is_boundary:
            self.forces += .0004*dir - .0008*self.velocities
        else:
            self.forces += .0007*dir - .0007*self.velocities

    @staticmethod
    def activation(t):
        m = 1.075
        k = .0215
        return t**m * math.e**(-k*t)

