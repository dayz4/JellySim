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
        # self.volume = self.calc_volume()
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

    # def calc_volume(self):
    #     return np.linalg.det(self.Dm()) / 6.0

    def calc_surface_area(self):
        v0, v1, v2, v3 = self.reference_vertices
        triangles = [[v0, v1, v2], [v0, v1, v3], [v0, v2, v3], [v1, v2, v3]]
        max_area = 0
        max_triangle = None
        for a, b, c in triangles:
            ab = b - a
            ac = c - a
            theta = math.acos(np.dot(ab, ac))
            area = np.linalg.norm(ab) * np.linalg.norm(ac) * math.cos(theta) / 2.0
            if area > max_area:
                max_area = area
                max_triangle = [a, b, c]
        return max_area, max_triangle

    def find_surface_triangle(self):
        v0, v1, v2, v3 = self.reference_vertices
        # return [v0, v1, v2], 0
        triangles = [[v0, v1, v2], [v0, v1, v3], [v0, v2, v3], [v1, v2, v3]]
        min_theta = math.pi
        surface_triangle = None
        surface_area = 0
        for a, b, c in triangles:
            ab = b - a
            ac = c - a
            normal = np.cross(ab, ac)
            theta = np.arccos(np.dot(normal, self.center) / (np.linalg.norm(normal) * np.linalg.norm(self.center)))
            theta = min(theta, math.pi - theta)
            if theta < min_theta:
                min_theta = theta
                surface_triangle = [a, b, c]
                surface_area = np.linalg.norm(ab) * np.linalg.norm(ac) * math.cos(theta) / 2.0
        return surface_triangle, surface_area

    def check_is_boundary(self):
        for v in self.reference_vertices:
            if v[1] < .01:
                return True
        return False

    def update(self, t, dt):
        if self.contracted and t - self.activation_time > 2:
            self.relax()
        if self.stiffened and t - self.stiffen_time > .8:
            self.unstiffen()
        self.compute_forces(t)

    def contract(self, strength, t):
        # jelly_center = np.array([0, 0, self.center[2]])
        # vertices_to_center = [jelly_center - v for v in self.reference_vertices]
        # vertices_to_center = [self.center - v for v in self.reference_vertices]
        # self.deformed_vertices = [
        #     self.reference_vertices[i] + (vertices_to_center[i] * .5 * strength) for i in range(len(self.reference_vertices))
        # ]
        # self.contracted_center = self.center + (jelly_center - self.center) * .5 * strength
        # self.reference_vertices, self.deformed_vertices = self.deformed_vertices, self.reference_vertices
        self.contracted = True
        self.activation_time = t

    def relax(self):
        # self.reference_vertices, self.deformed_vertices = self.deformed_vertices, self.reference_vertices
        self.contracted = False
        # self.contracted_center = None

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
            # if self.stiffened:
            #     F0 = .00005
            # else:
            #     F0 = .00003
            F0 = .00003
            # if self.is_boundary:
            #     F0 = .00003
            # F0 = .00003
            LF = np.linalg.norm(jelly_center - self.center)
            L0 = np.linalg.norm(self.center)
            S = 0.4
            F = F0 * math.e**(-((LF/L0 - 1) / S)**2)
            # F = .00003
            f = F * a

            # muscle_dir = resting_muscle_fiber / np.linalg.norm(resting_muscle_fiber)
            muscle_dir = resting_muscle_fiber / LF

            for i in range(4):
                self.forces[i] = f * muscle_dir
                continue
        # else:
        # pass
        time_elapsed = t - self.stiffen_time * 150
        a = self.activation(time_elapsed)
        # if self.stiffened:
        #     ks = max(.001, a*.008)
        # elif self.is_boundary:
        if self.is_boundary:
            ks = .001
        else:
            ks = .003

        # ks = .004
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
        # dldt = np.linalg.norm(self.velocities - self.velocities)
        dldt = 0
        f = ks * (1 - (RL / np.linalg.norm(d1d2, axis=0))) * d1d2 + bs * dldt

        for i, (i1, i2) in enumerate(spring_pairs):
            self.forces[i1] -= f[:, i]
            self.forces[i2] += f[:, i]

                # d1, d2 = self.deformed_vertices[i1], self.deformed_vertices[i2]
                # r1, r2 = self.reference_vertices[i1], self.reference_vertices[i2]

                # if np.linalg.norm(d1-r1) < .001 and np.linalg.norm(d2-r2) < .001:
                #     continue

                # if np.linalg.norm(self.velocities[i1]) < .01 and np.linalg.norm(self.velocities[i2]) < .01:
                #     continue

                # d1d2 = d1 - d2
                #
                # RL = np.linalg.norm(r1 - r2)
                # bs = .000025
                # # dldt = self.dldt[i]
                # dldt = np.linalg.norm(self.velocities[i1] - self.velocities[i2])

                # f = ks * (1 - (RL / np.linalg.norm(d1d2))) * d1d2 + bs * dldt

                # self.forces[i1] -= f
                # self.forces[i2] += f

        dir = np.subtract(self.reference_vertices, self.deformed_vertices)
        if self.is_boundary:
            # if self.stiffened:
            #     self.forces += .0003*dir - .0008*self.velocities
            # else:
            self.forces += .0004*dir - .0008*self.velocities
        else:
            self.forces += .0007*dir - .0007*self.velocities

        # for i, r in enumerate(self.reference_vertices):
        #     d = self.deformed_vertices[i]
        #     if np.linalg.norm(r - d) < .001:
        #         continue
        #     dir = (r - d)
        #     self.forces[i] += .001*dir - .001*self.velocities[i]

        # if (np.any(sum(self.forces) > 0)):
        #     print(self.forces)

    # def Dm(self):
    #     v1, v2, v3, v4 = self.reference_vertices
    #     return np.array([
    #         [v1[0] - v4[0], v2[0] - v4[0], v3[0] - v4[0]],
    #         [v1[1] - v4[1], v2[1] - v4[1], v3[1] - v4[1]],
    #         [v1[2] - v4[2], v2[2] - v4[2], v3[2] - v4[2]]
    #     ])
    #
    # def Ds(self):
    #     v1, v2, v3, v4 = self.deformed_vertices
    #     return np.array([
    #         [v1[0] - v4[0], v2[0] - v4[0], v3[0] - v4[0]],
    #         [v1[1] - v4[1], v2[1] - v4[1], v3[1] - v4[1]],
    #         [v1[2] - v4[2], v2[2] - v4[2], v3[2] - v4[2]]
    #     ])

    # def compute_forces(self):
    #     if self.contracted and np.linalg.det(self.Dm()) != 0:
    #         Dm_inv = np.linalg.inv(self.Dm())
    #         Ds = self.Ds()
    #         F = Ds @ Dm_inv
    #         # print(self.Dm())
    #         # print(Dm_inv)
    #         # print("F: ", F)
    #
    #         # P = 2µ*eps + λtr(eps)I
    #         epsilon = .001 #epsilon:Sym{δF} = epsilon:δF
    #         k = .00001
    #         nu = 1
    #         mu = k / (2 * 1 + nu)
    #         lambd = (k * nu) / ((1 + nu) * (1 - 2*nu))
    #         tr = 1 #I: Sym{δF} = I:δF
    #         I = np.identity(3)
    #         P = 2 * mu * epsilon + lambd * tr * (F - I) @ I
    #
    #         f1, f2, f3 = -self.volume * P @ np.transpose(Dm_inv)
    #         f4 = -f1 - f2 - f3
    #         self.forces = np.array([f1, f2, f3, f4])
    #         print(self.forces)
    #     else:
    #         self.forces = [0, 0, 0, 0]

    @staticmethod
    def activation(t):
        m = 1.075
        k = .0215
        # t in ms
        return t**m * math.e**(-k*t)

