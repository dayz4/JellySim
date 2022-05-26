from OpenGL.GL import *
from OpenGL.GLUT import *
import glm
import numpy as np
from food import Food
from shader import Shader
import random
from predator import Predator


class EnvMesh:
    def __init__(self, view, projection):
        self.view = view
        self.projection = projection
        self.food_count = 3
        self.food = self.init_food()
        # self.predator = self.init_predator()
        self.food_count = self.food.pos.shape[0]
        self.shader_program = Shader("shaders/vertex_env.glsl", "shaders/frag.glsl")

    def draw(self):
        self.food.draw(self.shader_program)
        # glColor3f(.4, .25, .15)
        # glBegin(GL_QUADS)
        # glVertex2f(0, 0)
        # glVertex2f(0, 1)
        # glVertex2f(.1, 0)
        # glVertex2f(.1, 1)
        # glEnd()
        # self.predator.draw(self.shader_program, view, self.projection)

    def init_food(self):
        food_pos = []
        for i in range(self.food_count):
            food_pos.append(self.generate_random_pos())

        # food_pos = [
        #     [5, 3, 2],
        #     [2, 5, -1],
        #     [4, -4, 1]
        # ]
        return Food(food_pos, self.view, self.projection)

    def init_predator(self):
        pos = self.generate_predator_pos()
        # pos = [-4, -2, -3]
        return Predator(pos)

    def respawn_food(self, food_idx):
        new_pos = np.array(self.generate_random_pos())
        self.food.respawn(food_idx, new_pos)

    def respawn_predator(self):
        self.predator = self.init_predator()

    def generate_random_pos(self):
        return [
            random.choice([random.random() * 3 - 6, random.random() * 3 + 3]),
            random.random() * 5 + 5,
            random.random() * 4 - 2
        ]

    def generate_predator_pos(self):
        return [
            random.random() * 8 - 4,
            random.random() * 3 - 5,
            random.random() * 6 - 3
        ]

    def world_pos(self):
        food_pos = self.food.world_pos()
        predator_pos = np.zeros(3)
        return np.concatenate((food_pos, predator_pos))

