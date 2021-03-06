from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np
import glm
import math
from shader import Shader


class Mesh:
    def __init__(self, vertices, normals, tetrahedrons, triangles, view, projection):
        self.vertices = vertices
        self.offsets = np.zeros(self.vertices.shape, dtype="float32")
        self.activations = np.zeros(self.vertices.shape[0], dtype="float32")
        self.normals = normals
        self.tetrahedrons = tetrahedrons
        self.triangles = triangles
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.pos = np.zeros(3)
        self.velocity = glm.vec3(0.0, 0.0, 0.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.rotation_axis = np.array([0.0, 1.0, 0.0])
        self.rotation_angle = 0
        self.current_rotation = 0
        self.rotation_start_time = 0
        self.rotating = False
        self.model = self.init_model_matrix()
        self.current_model = self.model
        self.shader_program = Shader("shaders/vertex.glsl", "shaders/frag.glsl")
        self.view = view
        self.projection = projection
        self.setup_buffers()

    def setup_buffers(self):
        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, 4*(self.vertices.size+self.offsets.size+self.activations.size), None, GL_DYNAMIC_DRAW)
        glBufferSubData(GL_ARRAY_BUFFER, 0, 4*self.vertices.size, self.vertices.flatten())
        glBufferSubData(GL_ARRAY_BUFFER, 4*self.vertices.size, 4*self.offsets.size, self.offsets.flatten())
        glBufferSubData(GL_ARRAY_BUFFER, 4*(self.vertices.size+self.offsets.size), 4 * self.activations.size, self.activations)

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*self.triangles.size, self.triangles.flatten(), GL_DYNAMIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*4, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*4, ctypes.c_void_p(4*self.vertices.size))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 4, ctypes.c_void_p(4*(self.vertices.size+self.offsets.size)))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

    def update_buffer_offsets(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 4*self.vertices.size, 4*self.offsets.size, self.offsets.flatten())
        glBufferSubData(GL_ARRAY_BUFFER, 4*(self.vertices.size+self.offsets.size), 4*self.activations.size, self.activations)

    def init_model_matrix(self):
        model = glm.mat4(1.0)
        return model

    def end_rotation(self, t):
        self.rotating = False
        self.model = self.current_model

    def world_pos(self):
        model = np.asarray(self.current_model)[:3, :3]
        world_pos = model @ self.pos
        return world_pos

    def add_translation(self, dq):
        self.model = glm.translate(self.model, glm.vec3(dq[0], dq[1], dq[2]))

    def update(self, t):
        if self.rotating:
            dt_total = t - self.rotation_start_time
            rotation = self.rotation_angle
            rotation *= max(0.0, 1 / (1 + math.exp(-2 * (dt_total - 1.5))) - .05)
            model = glm.rotate(self.model, rotation, glm.vec3(self.rotation_axis[0], self.rotation_axis[1], self.rotation_axis[2]))
            self.current_model = model
            if dt_total > 4:
                self.end_rotation(t)
        else:
            self.current_model = self.model

    def draw(self):
        self.update_buffer_offsets()
        self.shader_program.use()
        model = self.current_model

        self.shader_program.set_matrix("model", model)
        self.shader_program.set_matrix("view", self.view)
        self.shader_program.set_matrix("projection", self.projection)
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, len(self.triangles)*3, GL_UNSIGNED_INT, None)
