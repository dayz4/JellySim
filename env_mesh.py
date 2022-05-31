import numpy as np
from food import Food
from shader import Shader
import random


class EnvMesh:
    def __init__(self, view, projection):
        self.view = view
        self.projection = projection
        self.food_count = 3
        self.food = self.init_food()
        self.food_count = self.food.pos.shape[0]
        self.shader_program = Shader("shaders/vertex_env.glsl", "shaders/frag.glsl")

    def draw(self):
        self.food.draw(self.shader_program)

    def init_food(self):
        food_pos = []
        for i in range(self.food_count):
            food_pos.append(self.generate_random_pos())
        return Food(food_pos, self.view, self.projection)

    def respawn_food(self, food_idx):
        new_pos = np.array(self.generate_random_pos())
        self.food.respawn(food_idx, new_pos)

    def generate_random_pos(self):
        return [
            random.choice([random.random() * 3 - 6, random.random() * 3 + 3]),
            random.random() * 5 + 5,
            random.random() * 4 - 2
        ]
