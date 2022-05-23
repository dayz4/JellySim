from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np
import glm
import math


class Mesh:
    def __init__(self, vertices, normals, tetrahedrons, triangles):
        self.vertices = vertices
        self.offsets = np.zeros(self.vertices.shape, dtype="float32")
        self.activations = np.zeros(self.vertices.shape[0], dtype="float32")
        self.normals = normals
        self.tetrahedrons = tetrahedrons
        self.triangles = triangles
        self.vao = glGenVertexArrays(1)
        self.pos = glm.vec3(0.0, 0.0, 0.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.rotation_axis = np.array([0.0, 1.0, 0.0])
        self.rotation_angle = 0
        self.current_rotation = 0
        self.rotation_start_time = 0
        self.rotating = False
        self.model = self.init_model_matrix()
        self.current_model = self.model
        self.setup_buffers()

    def setup_buffers(self):
        glBindVertexArray(self.vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, 4*(self.vertices.size+self.offsets.size+self.activations.size), None, GL_DYNAMIC_DRAW)
        glBufferSubData(GL_ARRAY_BUFFER, 0, 4*self.vertices.size, self.vertices.flatten())
        glBufferSubData(GL_ARRAY_BUFFER, 4*self.vertices.size, 4*self.offsets.size, self.offsets.flatten())
        print(self.activations.flatten().shape)
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
        glBufferSubData(GL_ARRAY_BUFFER, 4*self.vertices.size, 4*self.offsets.size, self.offsets.flatten())
        glBufferSubData(GL_ARRAY_BUFFER, 4*(self.vertices.size+self.offsets.size), 4*self.activations.size, self.activations)

    def init_model_matrix(self):
        model = glm.mat4(1.0)
        # model = glm.rotate(model, glm.radians(-80.0), glm.vec3(1.0, 0.0, 0.0))
        # model = glm.rotate(model, glm.radians(-70.0), glm.vec3(0.0, 0.0, 1.0))
        return model

    def end_rotation(self, t):
        self.rotating = False
        self.model = self.current_model
        # rotation = self.rotation_angle
        # dt_total = t - self.rotation_start_time
        # rotation *= max(0, 1 / (1 + math.exp(-2 * (dt_total - 1.5))) - .05)
        #
        # self.model = glm.rotate(self.model, rotation, glm.vec3(self.rotation_axis[0], self.rotation_axis[1], self.rotation_axis[0]))

    def convert_to_world_coords(self):
        model = np.asarray(self.current_model)[:3, :3]
        pos = self.vertices + self.offsets
        world_pos = pos @ model
        return world_pos

    def draw(self, shader_program, t):
        self.update_buffer_offsets()

        shader_program.use()

        # print(dt_total / (dt_total + .5))

        dt_total = t - self.rotation_start_time
        if self.rotating and dt_total > 4:
            self.end_rotation(t)
        if self.rotating:
            rotation = self.rotation_angle
            rotation *= max(0, 1 / (1 + math.exp(-2 * (dt_total - 1.5))) - .05)
            self.last_rotation_time = t
            model = glm.rotate(self.model, rotation, glm.vec3(self.rotation_axis[0], self.rotation_axis[1], self.rotation_axis[0]))
            self.current_model = model
        else:
            model = self.model

        camera_pos = glm.vec3(0.0, 3.0, 10.0)

        view = glm.lookAt(camera_pos, self.pos, self.up)
        shader_program.set_matrix("view", view)

        projection = glm.mat4(glm.perspective(glm.radians(45.0), 1600 / 1600, 0.1, 100.0))

        shader_program.set_matrix("model", model)
        shader_program.set_matrix("view", view)
        shader_program.set_matrix("projection", projection)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, len(self.triangles)*12, GL_UNSIGNED_INT, None)
