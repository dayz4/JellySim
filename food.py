import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
import glm


class Food:
    def __init__(self, pos, view, projection):
        self.pos = np.array(pos)
        self.vertices = self.get_vertex_positions()
        self.offsets = self.get_offsets()
        self.triangles = self.get_triangles()
        self.velocities = np.zeros(self.pos.shape)
        self.model = self.set_model_matrix()
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.view = view
        self.projection = projection
        self.setup_buffers()

    def draw(self, shader_program):
        shader_program.use()

        shader_program.set_matrix("model", self.model)
        shader_program.set_matrix("view", self.view)
        shader_program.set_matrix("projection", self.projection)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, len(self.triangles)*3, GL_UNSIGNED_INT, None)

    def setup_buffers(self):
        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, 4*(self.vertices.size+self.offsets.size), None, GL_DYNAMIC_DRAW)
        glBufferSubData(GL_ARRAY_BUFFER, 0, 4*self.vertices.size, self.vertices.flatten())
        glBufferSubData(GL_ARRAY_BUFFER, 4*self.vertices.size, 4*self.offsets.size, self.offsets.flatten())

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*self.triangles.size, self.triangles.flatten(), GL_DYNAMIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*4, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*4, ctypes.c_void_p(4*self.vertices.size))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)

    def update_buffer_offsets(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 4*self.vertices.size, 4*self.offsets.size, self.offsets.flatten())

    def set_model_matrix(self):
        model = glm.mat4(1.0)
        model = glm.rotate(model, glm.radians(-45.0), glm.vec3(1.0, 1.0, 0.0))
        return model

    def get_vertex_positions(self):
        vertices = []
        for _ in self.pos:
            vertices.extend([
                [ .2,  .2,  .2],
                [ .2, -.2, -.2],
                [-.2,  .2, -.2],
                [-.2, -.2,  .2]
            ])
        return np.array(vertices, dtype="float32")

    def get_offsets(self):
        offsets = []
        for pos in self.pos:
            offsets.extend([pos, pos, pos, pos])
        return np.array(offsets, dtype="float32")

    def get_triangles(self):
        triangles = []
        for i in range(len(self.pos)):
            v0 = 0 + i*4
            v1 = 1 + i*4
            v2 = 2 + i*4
            v3 = 3 + i*4
            triangles.extend([[v0, v1, v2], [v0, v1, v3], [v0, v2, v3], [v1, v2, v3]])
        return np.array(triangles)

    def respawn(self, idx, pos):
        self.pos[idx] = pos
        self.offsets[idx * 4:idx * 4 + 4] = [pos, pos, pos, pos]
        self.update_buffer_offsets()


