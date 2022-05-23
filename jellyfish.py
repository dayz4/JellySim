import numpy as np
import collections
import pickle
import os
import math
import random


class Jellyfish:
    def __init__(self, mesh, muscles, radial_muscles, fem_elements, rhopalia):
        self.mesh = mesh
        self.muscles = muscles
        self.radial_muscles = radial_muscles
        self.fem_elements = fem_elements
        self.activations, self.innervations = self.load_muscle_activations()
        self.dnn_activations, self.dnn_innervations = self.load_dnn_activations()
        self.internal_forces = np.zeros((len(self.mesh.vertices), 3))
        self.positions = self.mesh.vertices
        self.velocities = np.zeros(self.positions.shape)
        self.nerve_net = self.load_nerve_net()
        self.dnn = self.load_nerve_net()
        self.last_mnn_conduction_time = 0
        self.last_dnn_conduction_time = 0
        self.conducting = False
        # self.conducting = True
        self.dnn_conducting = False
        self.new_nerve_activations = []
        # self.new_nerve_activations = [194]
        self.nerve_activations = [False] * len(self.mesh.vertices)
        self.dnn_nerve_activations = [False] * len(self.mesh.vertices)
        self.dnn_refractions = [False] * len(self.mesh.vertices)
        self.new_dnn_activations = []
        # self.new_dnn_activations = [194]
        self.last_activate_time = -7
        self.internal_positions = self.mesh.vertices
        self.internal_velocities = np.zeros(self.positions.shape)
        self.forces = np.zeros(self.positions.shape)
        self.mnn_delay = 1.4
        # self.dnn_activate_time = 0
        self.starting_nerve = None
        self.mnn_dnn_angle = None
        self.rhopalia = self.load_rhopalia(rhopalia)
        print(self.rhopalia)

    def draw(self, shader_program, t, dt):
        self.update(t, dt)
        self.mesh.draw(shader_program, t)

    def update(self, t, dt):
        if t - self.last_activate_time > 7:
            rhop = random.choice(self.rhopalia)
            self.activate_starting_radial_muscle(rhop, t)
            self.starting_nerve = rhop

        if self.dnn_conducting and not self.conducting and t - self.last_activate_time > self.mnn_delay:
            self.activate_starting_muscle(self.starting_nerve, t)

        if self.conducting and t - self.last_mnn_conduction_time > .01:
            self.propagate_nerve_signal(t)
            self.last_mnn_conduction_time = t
        if self.dnn_conducting and t - self.last_dnn_conduction_time > .1:
            self.propagate_dnn_signal(t)
            self.last_dnn_conduction_time = t

        if (self.conducting or self.dnn_conducting) and self.new_nerve_activations == [] and self.new_dnn_activations == []:
            self.conducting = False
            self.dnn_conducting = False
            # self.mesh.end_rotation(t)

        for fem_element in self.fem_elements:
            fem_element.update(t, dt)
        self.apply_forces(dt)

        for i, muscle in enumerate(self.radial_muscles):
            muscle.update(t)
            if not muscle.activated:
                activated_elements = self.dnn_activations[i]
                for fem_id, _ in activated_elements:
                    for vertex_id in self.fem_elements[fem_id].vertex_ids:
                        if self.nerve_activations[vertex_id]:
                            self.mesh.activations[vertex_id] = 1.0
                        else:
                            self.mesh.activations[vertex_id] = 0.0
                        if self.dnn_nerve_activations[vertex_id]:
                            self.dnn_nerve_activations[vertex_id] = False
                            self.dnn_refractions[vertex_id] = True
            if not muscle.refraction:
                activated_elements = self.dnn_activations[i]
                for fem_id, _ in activated_elements:
                    for vertex_id in self.fem_elements[fem_id].vertex_ids:
                        self.dnn_refractions[vertex_id] = False

        for i, muscle in enumerate(self.muscles):
            muscle.update(t)
            if not muscle.activated:
                activated_elements = self.activations[i]
                for fem_id, _ in activated_elements:
                    for vertex_id in self.fem_elements[fem_id].vertex_ids:
                        if self.dnn_nerve_activations[vertex_id] and self.dnn_innervations[vertex_id]:
                            self.mesh.activations[vertex_id] = 0.5
                        else:
                            self.mesh.activations[vertex_id] = 0.0
                        self.nerve_activations[vertex_id] = False

    def activate_starting_muscle(self, nerve, t):
        self.last_activate_time = t
        for muscle_idx in self.innervations[nerve]:
            self.activate_muscle(muscle_idx, t)
        self.new_nerve_activations = [nerve]
        self.conducting = True

    def activate_starting_radial_muscle(self, nerve, t):
        self.last_activate_time = t
        for muscle_idx in self.dnn_innervations[nerve]:
            self.activate_radial_muscle(muscle_idx, t)
        self.new_dnn_activations = [nerve]
        self.dnn_conducting = True

    def propagate_nerve_signal(self, t):
        new_nerve_activations = []
        for nerve in self.new_nerve_activations:
            for neighbor in self.nerve_net[nerve]:
                if not self.nerve_activations[neighbor]:
                    for muscle_idx in self.innervations[neighbor]:
                        self.activate_muscle(muscle_idx, t)
                    new_nerve_activations.append(neighbor)
                    self.nerve_activations[neighbor] = True

                    for neighbor2 in self.nerve_net[neighbor]:
                        if not self.nerve_activations[neighbor2]:
                            for muscle_idx in self.innervations[neighbor2]:
                                self.activate_muscle(muscle_idx, t)
                            new_nerve_activations.append(neighbor2)
                            self.nerve_activations[neighbor2] = True

                            # for neighbor3 in self.nerve_net[neighbor2]:
                            #     if not self.nerve_activations[neighbor3]:
                            #         for muscle_idx in self.innervations[neighbor3]:
                            #             self.activate_muscle(muscle_idx, t)
                            #         new_nerve_activations.append(neighbor3)
                            #         self.nerve_activations[neighbor3] = True
        self.new_nerve_activations = new_nerve_activations

    def propagate_dnn_signal(self, t):
        new_dnn_activations = []
        for nerve in self.new_dnn_activations:
            for neighbor in self.dnn[nerve]:
                if not self.dnn_nerve_activations[neighbor] and not self.dnn_refractions[neighbor]:
                    for muscle_idx in self.dnn_innervations[neighbor]:
                        self.activate_radial_muscle(muscle_idx, t)
                    new_dnn_activations.append(neighbor)
                    self.dnn_nerve_activations[neighbor] = True
        self.new_dnn_activations = new_dnn_activations

    def aggregate_internal_forces(self):
        internal_forces = np.zeros((len(self.mesh.vertices), 3))
        counts = np.zeros((len(self.mesh.vertices), 1))
        for fem_element in self.fem_elements:
            v1, v2, v3, v4 = fem_element.vertex_ids
            internal_forces[v1] += fem_element.forces[0]
            internal_forces[v2] += fem_element.forces[1]
            internal_forces[v3] += fem_element.forces[2]
            internal_forces[v4] += fem_element.forces[3]
            counts[v1] += 1
            counts[v2] += 1
            counts[v3] += 1
            counts[v4] += 1
        # internal_forces /= counts
        return internal_forces

    @staticmethod
    def euler(qt, vt, fint, fext, h):
        mass = .005
        v_next = vt + h * (fint + fext) / mass
        q_next = qt + h * v_next

        # v_next = (mass * vt + h * (fint + fext)) / (mass - h**2 * dfdt)

        return q_next, v_next

    def apply_forces(self, dt):
        normals = self.calc_normals()
        vs = np.zeros((len(self.fem_elements), 3))
        for i, fem_element in enumerate(self.fem_elements):
            v0, v1, v2, v3 = fem_element.vertex_ids
            v = (self.velocities[v0] + self.velocities[v1] + self.velocities[v2] + self.velocities[v3]) / 4.0
            vs[i] = v
        vnorms = np.linalg.norm(vs, axis=1)

        thrust = self.compute_thrust(normals, vnorms, vs) / 10.0
        # drag = -self.compute_drag(normals, vnorms, vs) / 10000.0
        # thrust = 0
        drag = 0
        qt = self.positions
        vt = self.velocities
        fint = self.aggregate_internal_forces()
        fext = thrust + drag

        self.apply_internal_forces(fint, dt)

        # fint = np.zeros(fint.shape)
        # _, v_next = self.euler(qt, vt, np.zeros(3), fext, dt)
        # avg_v = np.mean(self.velocities, axis=0)
        # vnorm = np.linalg.norm(avg_v)
        # if vnorm > 0:
        #     dir = avg_v / vnorm
        #     self.mesh.rotation_angle = math.acos(np.dot(avg_v, self.mesh.up))
        #     self.mesh.rotation_axis = np.cross(dir, self.mesh.up)
        #     # self.mesh.up = self.mesh.up * .5 + dir * .5
        #     self.mesh.up = dir

        q_next, v_next = self.euler(qt, vt, fint, fext, dt)
        self.mesh.offsets += q_next - qt
        self.positions = q_next
        # for i, v in enumerate(v_next):
        #     if sum(v) < .0001:
        #         v_next[i] = 0
        self.velocities = v_next



        # self.mesh.get_rotation(angle, axis)

        # print(avg_v)
        # max_v = np.max(self.velocities, axis=0)
        # if np.linalg.norm(max_v) > 0:
        #     self.velocities += (1 - self.velocities / max_v) * avg_v
        # for i, v in enumerate(self.velocities):
        #     if np.linalg.norm(v) < .1:
        # self.smooth_velocities()

    def apply_internal_forces(self, fint, dt):
        qt = self.internal_positions
        vt = self.internal_velocities
        q_next, v_next = self.euler(qt, vt, fint, np.zeros(3), dt)

        for fem_element in self.fem_elements:
            v1, v2, v3, v4 = fem_element.vertex_ids
            # d1, d2, d3, d4 = fem_element.deformed_vertices
            # lengths = np.array([
            #     (d2 - d1) - (q_next[v2] - q_next[v1]),
            #     (d3 - d1) - (q_next[v3] - q_next[v1]),
            #     (d4 - d1) - (q_next[v4] - q_next[v1]),
            #     (d3 - d2) - (q_next[v3] - q_next[v2]),
            #     (d4 - d2) - (q_next[v4] - q_next[v2]),
            #     (d4 - d3) - (q_next[v4] - q_next[v3])
            # ])
            # dl = np.linalg.norm(lengths)
            # fem_element.dldt = dl / (dt * 100)

            # fem_element.deformed_vertices[0] = q_next[v1]
            # fem_element.deformed_vertices[1] = q_next[v2]
            # fem_element.deformed_vertices[2] = q_next[v3]
            # fem_element.deformed_vertices[3] = q_next[v4]
            #
            # fem_element.velocities[0] = v_next[v1]
            # fem_element.velocities[1] = v_next[v2]
            # fem_element.velocities[2] = v_next[v3]
            # fem_element.velocities[3] = v_next[v4]

            fem_element.deformed_vertices =\
                np.array([q_next[v1], q_next[v2], q_next[v3], q_next[v4]])
            fem_element.velocities =\
                np.array([v_next[v1], v_next[v2], v_next[v3], v_next[v4]])

        self.internal_velocities = v_next
        self.internal_positions = q_next

    def compute_thrust(self, normals, vnorms, vs):
        # overall_thrust = np.zeros((len(self.mesh.vertices), 3))
        total_thrust = np.zeros(3)
        for i, fem_element in enumerate(self.fem_elements):
            # v0, v1, v2, v3 = fem_element.vertex_ids
            # v = (self.velocities[v0] + self.velocities[v1] + self.velocities[v2] + self.velocities[v3]) / 4.0
            # vnorm = np.linalg.norm(v)
            if vnorms[i] == 0:
                continue
            A = fem_element.surface_area
            n = normals[i]
            p = 1000 # density
            v = vs[i]
            # postheta = math.acos(np.dot(n, v))
            # negtheta = math.acos(np.dot(-n, v))
            # if postheta < negtheta:
            #     phi = math.pi / 2.0 - postheta
            #     C = self.C_thrust(phi)
            #     thrust = -p * A * C * vnorms[i] ** 2 * n / 2.0
            #     print(thrust)
            # else:
            #     phi = math.pi / 2.0 - negtheta
            #     C = self.C_thrust(phi)
            #     thrust = -p * A * C * vnorms[i] ** 2 * -n / 2.0
            #     print(thrust)
            # theta = max(math.acos(np.dot(n, v)), math.acos(np.dot(-n, v)))
            phi = math.pi / 2.0 - math.acos(np.dot(n, v))
            C = self.C_thrust(phi)
            thrust = -p * A * C * vnorms[i]**2 * n / 2.0
            total_thrust += thrust
            # thrust *= 20
            # overall_thrust[v0] += thrust
            # overall_thrust[v1] += thrust
            # overall_thrust[v2] += thrust
            # overall_thrust[v3] += thrust
        return total_thrust

    def calc_normals(self):
        ba = np.zeros((len(self.fem_elements), 3))
        ca = np.zeros((len(self.fem_elements), 3))
        for i, fem_element in enumerate(self.fem_elements):
            a, b, c = fem_element.surface_triangle
            ba[i] = b - a
            ca[i] = c - a
        normals = np.cross(ba, ca)
        return normals

    @staticmethod
    def C_thrust(phi):
        return 0.25 * (math.exp(0.8 * phi) - 1)

    def compute_drag(self, normals, vnorms, vs):
        total_drag = np.zeros(3)
        for i, fem_element in enumerate(self.fem_elements):
            v = vs[i]
            vnorm = vnorms[i]
            if vnorm == 0:
                continue
            A = fem_element.surface_area
            d = v / vnorm
            n = normals[i]
            p = 1000 # density
            phi = math.pi / 2.0 - math.acos(np.dot(-n, v))
            C = self.C_drag(phi)
            drag = p * A * C * vnorm**2 * d / 2.0
            total_drag += drag
        return total_drag

    @staticmethod
    def C_drag(phi):
        return -math.cos(phi * 2.0) + 1.05

    def set_mesh_rotation(self, fem_element, t):
        starting_pos = self.mesh.vertices[self.starting_nerve]
        start_dir = starting_pos / np.linalg.norm(starting_pos)
        mnn_dnn_angle = math.acos(np.dot(start_dir, fem_element.center / np.linalg.norm(fem_element.center)))

        # self.mesh.rotation_angle = self.mesh.current_rotation + (math.cos(2 * mnn_dnn_angle)) / 2.0
        self.mesh.rotation_angle = -math.cos(mnn_dnn_angle) * .5
        self.mesh.rotation_axis = np.cross(start_dir, np.array([0.0, 1.0, 0.0]))
        self.mesh.rotation_start_time = t
        self.mesh.rotating = True
        # self.mesh.up = self.mesh.up * .5 + dir * .5
        # self.mesh.up = dir
        # self.mesh.rotation_angle
        # print("set angle to", self.mesh.rotation_angle)

    def activate_muscle(self, muscle_idx, t):
        if self.muscles[muscle_idx].activated:
            return
        # print("activating", muscle_idx)
        activated_elements = self.activations[muscle_idx]
        for fem_id, dist in activated_elements:
            for vertex_id in self.fem_elements[fem_id].vertex_ids:
                self.mesh.activations[vertex_id] = 1.0
            strength = math.e ** (-10 * dist)
            self.fem_elements[fem_id].contract(strength, t)
            if not self.mesh.rotating and self.fem_elements[fem_id].stiffened:
                self.set_mesh_rotation(self.fem_elements[fem_id], t)
        self.muscles[muscle_idx].activated = True
        self.muscles[muscle_idx].activate_time = t

    def activate_radial_muscle(self, muscle_idx, t):
        if self.radial_muscles[muscle_idx].activated or self.radial_muscles[muscle_idx].refraction:
            return
        activated_elements = self.dnn_activations[muscle_idx]
        for fem_id, dist in activated_elements:
            for vertex_id in self.fem_elements[fem_id].vertex_ids:
                self.mesh.activations[vertex_id] = .5
            self.fem_elements[fem_id].stiffen(t)
        self.radial_muscles[muscle_idx].activated = True
        self.radial_muscles[muscle_idx].activate_time = t

    def compute_muscle_activations(self):
        activations = collections.defaultdict(list)
        innervations = collections.defaultdict(set)
        for fem_id, fem_element in enumerate(self.fem_elements):
            for idx, muscle in enumerate(self.muscles):
                dist = calc_line_point_dist(muscle.p1, muscle.p2, fem_element.center)
                if dist < 0:
                    continue
                if dist < .12:
                    activations[idx].append([fem_id, dist])
                    for v in fem_element.vertex_ids:
                        innervations[v].add(idx)
        return activations, innervations

    def load_muscle_activations(self):
        recompute = False

        if os.path.isfile('precomputed/muscle_activations.pickle') and not recompute:
            with open('precomputed/muscle_activations.pickle', 'rb') as f:
                return pickle.load(f)
        else:
            activations = self.compute_muscle_activations()
            self.save_muscle_activations(activations)
            return activations

    @staticmethod
    def save_muscle_activations(activations):
        with open('precomputed/muscle_activations.pickle', 'wb') as f:
            pickle.dump(activations, f, pickle.HIGHEST_PROTOCOL)

    def compute_dnn_activations(self):
        activations = collections.defaultdict(list)
        innervations = collections.defaultdict(set)
        for fem_id, fem_element in enumerate(self.fem_elements):
            for idx, muscle in enumerate(self.radial_muscles):
                dist = calc_line_point_dist(muscle.p1, muscle.p2, fem_element.center)
                if dist < 0:
                    continue
                if dist < .12:
                    activations[idx].append([fem_id, dist])
                    for v in fem_element.vertex_ids:
                        innervations[v].add(idx)
        return activations, innervations

    def load_dnn_activations(self):
        recompute = False

        if os.path.isfile('precomputed/dnn_activations.pickle') and not recompute:
            with open('precomputed/dnn_activations.pickle', 'rb') as f:
                return pickle.load(f)
        else:
            activations = self.compute_dnn_activations()
            self.save_dnn_activations(activations)
            return activations

    @staticmethod
    def save_dnn_activations(activations):
        with open('precomputed/dnn_activations.pickle', 'wb') as f:
            pickle.dump(activations, f, pickle.HIGHEST_PROTOCOL)

    def compute_nerve_net(self):
        adj_list = collections.defaultdict(set)
        for v1, v2, v3 in self.mesh.triangles:
            adj_list[v1].update([v2, v3])
            adj_list[v2].update([v1, v3])
            adj_list[v3].update([v1, v2])
        return adj_list

    def load_nerve_net(self):
        recompute = False

        if os.path.isfile('precomputed/nerve_net.pickle') and not recompute:
            with open('precomputed/nerve_net.pickle', 'rb') as f:
                return pickle.load(f)
        else:
            nerve_net = self.compute_nerve_net()
            self.save_nerve_net(nerve_net)
            return nerve_net

    @staticmethod
    def save_nerve_net(nerve_net):
        with open('precomputed/nerve_net.pickle', 'wb') as f:
            pickle.dump(nerve_net, f, pickle.HIGHEST_PROTOCOL)

    def find_rhopalia(self, rhopalia_pos):
        rhopalia = []
        for pos in rhopalia_pos:
            min_dist = 1
            closest = None
            for i, v in enumerate(self.mesh.vertices):
                dist = np.linalg.norm(pos - v)
                if dist < min_dist:
                    min_dist = dist
                    closest = i
            rhopalia.append(closest)
        return rhopalia

    def load_rhopalia(self, rhopalia_pos):
        recompute = False

        if os.path.isfile('precomputed/rhopalia.pickle') and not recompute:
            with open('precomputed/rhopalia.pickle', 'rb') as f:
                return pickle.load(f)
        else:
            rhopalia = self.find_rhopalia(rhopalia_pos)
            self.save_rhopalia(rhopalia)
            return rhopalia

    @staticmethod
    def save_rhopalia(rhopalia):
        with open('precomputed/rhopalia.pickle', 'wb') as f:
            pickle.dump(rhopalia, f, pickle.HIGHEST_PROTOCOL)

    # def smooth_velocities(self):
    #     smoothed_velocities = np.zeros(self.velocities.shape)
    #     for i, v in enumerate(self.velocities):
    #         neighbors = self.nerve_net[i]
    #         avg_neighbors = np.zeros(3)
    #         for neighbor in neighbors:
    #             avg_neighbors += self.velocities[neighbor]
    #         avg_neighbors /= len(neighbors)
    #
    #         if np.linalg.norm(avg_neighbors - v) > .01:
    #             # print(v, avg_neighbors)
    #             self.velocities[i] = avg_neighbors
    #         # smoothed_velocities[i] = avg_neighbors
    #     # self.velocities = smoothed_velocities

    # def contract(self, fem_element, muscle, dist):
    #     muscle_dir = muscle.p1 - muscle.p2
    #     muscle_dir = muscle_dir / np.linalg.norm(muscle_dir)
    #     print(muscle_dir)
    #
    #     strength = math.e**(-10*dist)
    #     print(1-strength)
    #
    #     centered = np.array(fem_element.reference_vertices) - fem_element.center
    #     print(centered)
    #     centered *= np.ones(muscle_dir.shape) - (muscle_dir * (1.0 - strength))
    #     print(centered)
    #
    #     #     [
    #     #     tetrahedron.p1 - tetrahedron.center,
    #     #     tetrahedron.p2 - tetrahedron.center,
    #     #     tetrahedron.p3 - tetrahedron.center,
    #     #     tetrahedron.p4 - tetrahedron.center
    #     # ]

    # def update_fem_deformation(self):
    #     for fem_element in self.fem_elements:
    #         v1, v2, v3, v4 = fem_element.vertex_ids
    #         fem_element.deformed_vertices =\
    #             np.array([self.positions[v1], self.positions[v2], self.positions[v3], self.positions[v4]])
    #         fem_element.velocities =\
    #             np.array([self.velocities[v1], self.velocities[v2], self.velocities[v3], self.velocities[v4]])


def calc_line_point_dist(a, b, p):
    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # nonzero means that point is not within bounds of line segment
    if h > 0:
        return -1

    # perpendicular vector
    c = np.cross(p - a, d)
    return np.linalg.norm(c)

    # ab = b - a
    # av = p - a
    # bv = p - b
    #
    # if np.dot(av, ab) <= 0 or np.dot(bv, ab) >= 0:
    #     return -1
    #
    # # calc perpendicular distance
    # cross = np.cross(ab, av)
    # return np.sqrt(np.dot(cross, cross)) / np.sqrt(np.dot(ab, ab))

