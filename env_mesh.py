import numpy as np
from food import Food
from shader import Shader
import random
from predator import Predator


class EnvMesh:
    def __init__(self, view, projection):
        self.food_count = 3
        self.food = self.init_food()
        self.predator = self.init_predator()
        self.food_count = self.food.pos.shape[0]
        self.shader_program = Shader("shaders/vertex_env.glsl", "shaders/frag.glsl")
        self.view = view
        self.projection = projection

    def draw(self):
        self.food.draw(self.shader_program, self.view, self.projection)
        self.predator.draw(self.shader_program, self.view, self.projection)

    def init_food(self):
        food_pos = []
        for i in range(self.food_count):
            food_pos.append(self.generate_random_pos())

        # food_pos = [
        #     [5, 3, 2],
        #     [2, 5, -1],
        #     [4, -4, 1]
        # ]
        return Food(food_pos)

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
            random.random() * 10 - 5,
            random.random() * 3 + 2,
            random.random() * 10 - 5
        ]

    def generate_predator_pos(self):
        return [
            random.random() * 2 - 5,
            random.random() * 3 - 5,
            random.random() * 10 - 5
        ]

    def world_pos(self):
        food_pos = self.food.world_pos()
        predator_pos = np.zeros(3)
        return np.concatenate((food_pos, predator_pos))

