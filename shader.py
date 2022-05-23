import glm
from OpenGL.GL import *


class Shader:
    def __init__(self, vertex_file, frag_file):
        self.vertex_shader = self.read_file(vertex_file)
        self.frag_shader = self.read_file(frag_file)
        self.id = glCreateProgram()
        self.link()

    def link(self):
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, self.vertex_shader)
        glCompileShader(vertex_shader)
        success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
        if not success:
            infolog = glGetShaderInfoLog(vertex_shader)
            print(infolog)

        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, self.frag_shader)
        glCompileShader(fragment_shader)
        success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
        if not success:
            infolog = glGetShaderInfoLog(fragment_shader)
            print(infolog)

        glAttachShader(self.id, vertex_shader)
        glAttachShader(self.id, fragment_shader)
        glLinkProgram(self.id)
        success = glGetProgramiv(self.id, GL_LINK_STATUS)
        if not success:
            infolog = glGetProgramInfoLog(self.id)
            print(infolog)

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

    def use(self):
        glUseProgram(self.id)

    def set_matrix(self, name, matrix):
        matrix_loc = glGetUniformLocation(self.id, name)
        glUniformMatrix4fv(matrix_loc, 1, GL_FALSE, glm.value_ptr(matrix))

    @staticmethod
    def read_file(fn):
        f = open(fn, "r")
        return f.read()

