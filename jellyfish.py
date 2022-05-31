import numpy as np
import collections
import pickle
import os
import math
import random

import scipy
from scipy import spatial


class Jellyfish:
    def __init__(self, mesh, muscles, radial_muscles, fem_elements, rhopalia):
        self.mesh = mesh
        self.muscles = muscles
        self.radial_muscles = radial_muscles
        self.fem_elements = fem_elements
        self.activations, self.innervations = self.load_muscle_activations()
        self.dnn_activations, self.dnn_innervations = self.load_dnn_activations()
        self.internal_forces = np.zeros((len(self.mesh.vertices), 3))
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.nerve_net = self.load_nerve_net()
        self.dnn = self.load_nerve_net()
        self.last_mnn_conduction_time = 0
        self.last_dnn_conduction_time = 0
        self.conducting = False
        self.dnn_conducting = False
        self.new_nerve_activations = []
        self.nerve_activations = [False] * len(self.mesh.vertices)
        self.dnn_nerve_activations = [False] * len(self.mesh.vertices)
        self.dnn_refractions = [False] * len(self.mesh.vertices)
        self.new_dnn_activations = []
        self.last_activate_time = -7
        self.internal_positions = self.mesh.vertices
        self.internal_velocities = np.zeros(self.internal_positions.shape)
        self.forces = np.zeros(self.internal_positions.shape)
        self.mnn_delay = 0
        self.starting_nerve = None
        self.mnn_dnn_angle = None
        self.rhopalia = self.load_rhopalia(rhopalia)
        self.at_rest = True
        self.rotation_start_time = 0
        self.rotation_angle = 0
        self.rotation_axis = None
        self.rotating = False

    def draw(self):
        self.mesh.draw()

    def update(self, t, dt, rhop_idx, mnn_delay):
        if t - self.last_activate_time > 6:
            self.at_rest = True
        if self.at_rest:
            self.mnn_delay = mnn_delay
            rhop = self.rhopalia[rhop_idx]
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

        for fem_element in self.fem_elements:
            fem_element.update(t)
        self.apply_forces(t, dt)

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

        self.mesh.update(t)

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
        self.at_rest = False

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
        for fem_element in self.fem_elements:
            v1, v2, v3, v4 = fem_element.vertex_ids
            internal_forces[v1] += fem_element.forces[0]
            internal_forces[v2] += fem_element.forces[1]
            internal_forces[v3] += fem_element.forces[2]
            internal_forces[v4] += fem_element.forces[3]
        return internal_forces

    @staticmethod
    def euler(qt, vt, fint, fext, h):
        mass = .005
        v_next = vt + h * (fint + fext) / mass
        q_next = qt + h * v_next
        return q_next, v_next

    def apply_forces(self, t, dt):
        normals = self.calc_normals()
        vs = np.zeros((len(self.fem_elements), 3))
        for i, fem_element in enumerate(self.fem_elements):
            v0, v1, v2, v3 = fem_element.vertex_ids
            v = (self.internal_velocities[v0] + self.internal_velocities[v1] + self.internal_velocities[v2] + self.internal_velocities[v3]) / 4.0
            vs[i] = v
        vnorms = np.linalg.norm(vs, axis=1)

        thrust = self.compute_thrust(normals, vnorms, vs) / 10.0
        drag = -self.compute_drag(normals, self.velocity) / 60000.0
        vt = self.velocity
        fint = self.aggregate_internal_forces()
        fext = thrust + drag

        self.apply_internal_forces(fint, dt, t)

        dq, v_next = self.euler(np.zeros(3), vt, 0, fext, dt)
        self.velocity = v_next
        if self.rotating:
            dt_total = t - self.rotation_start_time
            rotation = self.rotation_angle
            rotation *= max(0.0, 1 / (1 + math.exp(-2 * (dt_total - 1.5))) - .05)
            rot_vec = self.rotation_axis * rotation
            rotation_matrix = scipy.spatial.transform.Rotation.from_rotvec(rot_vec)
            dq = rotation_matrix.apply(dq)
            v_next = rotation_matrix.apply(dq)
            if dt > 4:
                self.rotating = False

        self.position += dq
        self.mesh.add_translation(dq)
        self.mesh.pos = self.position
        self.mesh.velocity = v_next

    def apply_internal_forces(self, fint, dt, t):
        qt = self.internal_positions
        vt = self.internal_velocities
        q_next, v_next = self.euler(qt, vt, fint, 0, dt)

        for fem_element in self.fem_elements:
            v1, v2, v3, v4 = fem_element.vertex_ids
            fem_element.deformed_vertices =\
                np.array([q_next[v1], q_next[v2], q_next[v3], q_next[v4]])
            fem_element.velocities =\
                np.array([v_next[v1], v_next[v2], v_next[v3], v_next[v4]])

        self.mesh.offsets += q_next - qt
        self.internal_velocities = v_next
        self.internal_positions = q_next

    def compute_thrust(self, normals, vnorms, vs):
        total_thrust = np.zeros(3)
        for i, fem_element in enumerate(self.fem_elements):
            if vnorms[i] == 0:
                continue
            A = fem_element.surface_area
            n = normals[i]
            p = 1000 # density
            v = vs[i]
            phi = math.pi / 2.0 - math.acos(np.dot(n, v))
            C = self.C_thrust(phi)
            thrust = -p * A * C * vnorms[i]**2 * n / 2.0
            total_thrust += thrust
        return total_thrust

    def calc_normals(self):
        ba = np.zeros((len(self.fem_elements), 3))
        ca = np.zeros((len(self.fem_elements), 3))
        for i, fem_element in enumerate(self.fem_elements):
            v1, v2, v3 = fem_element.surface_triangle
            a = self.mesh.vertices[v1] + self.mesh.offsets[v1]
            b = self.mesh.vertices[v2] + self.mesh.offsets[v2]
            c = self.mesh.vertices[v3] + self.mesh.offsets[v3]
            ba[i] = b - a
            ca[i] = c - a
        normals = np.cross(ba, ca)
        return normals

    @staticmethod
    def C_thrust(phi):
        return 0.25 * (math.exp(0.8 * phi) - 1)

    def compute_drag(self, normals, v):
        total_drag = np.zeros(3)
        for i, fem_element in enumerate(self.fem_elements):
            vnorm = np.linalg.norm(v)
            if vnorm == 0:
                continue
            A = fem_element.surface_area
            d = v / vnorm
            n = normals[i]
            p = 1000 # density
            phi = math.pi / 2.0 - math.acos(np.dot(n, v))
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

        rotation_axis = np.cross(start_dir, np.array([0.0, 1.0, 0.0]))
        rotation_angle = -math.cos(mnn_dnn_angle) * .5

        self.mesh.rotation_angle = rotation_angle
        self.mesh.rotation_axis = rotation_axis
        self.mesh.rotation_start_time = t
        self.mesh.rotating = True
        self.rotation_start_time = t
        self.rotation_axis = rotation_axis
        self.rotating = True
        self.rotation_angle = rotation_angle

    def activate_muscle(self, muscle_idx, t):
        if self.muscles[muscle_idx].activated:
            return
        activated_elements = self.activations[muscle_idx]
        for fem_id, dist in activated_elements:
            for vertex_id in self.fem_elements[fem_id].vertex_ids:
                self.mesh.activations[vertex_id] = 1.0
            self.fem_elements[fem_id].contract(t)
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


def calc_line_point_dist(a, b, p):
    d = np.divide(b - a, np.linalg.norm(b - a))

    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    h = np.maximum.reduce([s, t, 0])

    if h > 0:
        return -1

    c = np.cross(p - a, d)
    return np.linalg.norm(c)
