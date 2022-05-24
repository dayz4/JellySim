import numpy as np
from food import Food
from shader import Shader
import random


class EnvMesh:
    def __init__(self):
        self.food_count = 2
        self.food = self.init_food()
        self.predator = None
        self.food_count = self.food.pos.shape[0]
        self.shader_program = Shader("shaders/vertex_env.glsl", "shaders/frag.glsl")

    def draw(self, view, projection):
        self.food.draw(self.shader_program, view, projection)

    def init_food(self):
        food_pos = []
        for i in range(self.food_count):
            food_pos.append(self.generate_random_pos())
        return Food(food_pos)

    def respawn_food(self, food_idx):
        new_pos = np.array(self.generate_random_pos())
        self.food.respawn(food_idx, new_pos)

    def generate_random_pos(self):
        return [
            random.random() * 10 - 5,
            random.random() * 10 - 5,
            random.random() * 10 - 5
        ]

    def world_pos_velocity(self):
        food_pos, food_vel = self.food.world_pos_velocity()
        predator_pos, predator_vel = np.zeros(3), np.zeros(3)
        return np.concatenate((food_pos, predator_pos)), np.concatenate((food_vel, predator_vel))

