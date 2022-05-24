import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
import glm


class Predator:
    def __init__(self, pos):
        self.pos = pos
        self.vertices = self.get_vertex_positions()
        self.offsets = self.get_offsets()
        self.triangles = self.get_triangles()
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.model = self.set_model_matrix()
        self.velocity_magnitude = .1
        self.setup_buffers()

    def draw(self, shader_program, view, projection):
        self.update_buffer_offsets()

        shader_program.use()

        shader_program.set_matrix("model", self.model)
        shader_program.set_matrix("view", view)
        shader_program.set_matrix("projection", projection)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, len(self.triangles)*3, GL_UNSIGNED_INT, None)

    def update(self, jelly_pos, dt):
        to_jelly = jelly_pos - self.pos
        dir = to_jelly / np.linalg.norm(to_jelly)
        self.pos += dir * self.velocity_magnitude * dt
        self.offsets = self.get_offsets()

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
        # model = glm.rotate(model, glm.radians(-45.0), glm.vec3(1.0, 1.0, 0.0))
        return model

    def get_vertex_positions(self):
        vertices = [
            [ .2,  .2,  .2],
            [ .2, -.2,  .2],
            [ .2,  .2, -.2],
            [ .2, -.2, -.2],
            [-.2,  .2,  .2],
            [-.2, -.2,  .2],
            [-.2,  .2, -.2],
            [-.2, -.2, -.2]
        ]
        return np.array(vertices, dtype="float32")

    def get_offsets(self):
        offsets = [self.pos, self.pos, self.pos, self.pos, self.pos, self.pos, self.pos, self.pos]
        return np.array(offsets, dtype="float32")

    def get_triangles(self):
        triangles = [
            [0, 1, 2],
            [1, 3, 2],
            [0, 2, 6],
            [0, 6, 4],
            [4, 6, 7],
            [5, 4, 7],
            [1, 5, 3],
            [3, 5, 7],
            [0, 5, 1],
            [0, 4, 5],
            [2, 3, 7],
            [2, 6, 7]
        ]
        return np.array(triangles)

    def world_pos(self):
        return self.pos