import gym
import numpy as np
import glfw
from OpenGL.GL import *
import glm
from shader import Shader
import model_loader
from env_mesh import EnvMesh


class OceanEnv(gym.Env):
    def __init__(self):
        super(OceanEnv, self).__init__()

        self.width, self.height = 1600, 1600

        self.last_frame = 0
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.window = self.init_render()
        self.jellyfish, self.env_mesh = self.init_models()

        obs_dim = self.get_state().size
        pos_bound = np.zeros(obs_dim)
        pos_bound.fill(10)
        self.observation_space = gym.spaces.Box(
            low=-pos_bound,
            high=pos_bound,
            dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            low=0,
            high=2,
            shape=(8,),
            dtype=np.float64
        )

        self.reward = 0
        self.total_dist_from_food, self.dist_from_closest_food, self.closest_food_idx = self.calc_dist_from_food()
        # self.view = self.get_view_matrix()
        # self.dist_from_predator = self.calc_dist_from_predator()

    def reset(self):
        self.jellyfish, self.env_mesh = self.init_models()
        self.total_dist_from_food, self.dist_from_closest_food, self.closest_food_idx = self.calc_dist_from_food()
        # self.dist_from_predator = self.calc_dist_from_predator()
        self.reward = 0
        return self.get_state()

    def step(self, action):
        # print("action", action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # print(action)
        current_frame = glfw.get_time() / 3.0
        dt = current_frame - self.last_frame
        # print(dt)

        # rhop_idx = action[0]
        # mnn_delay = action[1][rhop_idx]
        # rhop_idx = max(range(len(action)), key=values.__getitem__)
        rhop_idx = np.argmax(action)
        mnn_delay = np.max(action)
        self.jellyfish.update(current_frame, dt, rhop_idx, mnn_delay)
        # self.env_mesh.predator.update(self.jellyfish.mesh.world_pos(), dt)
        self.last_frame = current_frame

        total_dist_from_food, dist_from_closest_food, closest_food_idx = self.calc_dist_from_food()

        # if total_dist_from_food-.01 > self.total_dist_from_food:
        #     # print("A")
        #     self.reward -= 5 * abs(self.total_dist_from_food - total_dist_from_food)
        # if total_dist_from_food+.01 < self.total_dist_from_food:
            # print("B")
        # self.reward += 1 * (self.total_dist_from_food - total_dist_from_food)

        # if dist_from_closest_food-.01 > self.dist_from_closest_food:
        #     # print("C")
        #     self.reward -= 2.5 * abs(self.dist_from_closest_food - dist_from_closest_food)
        # if dist_from_closest_food+.01 < self.dist_from_closest_food:
            # print("D")
        self.reward += 1.5 * (self.dist_from_closest_food - dist_from_closest_food)
        # print(dist_from_closest_food)
        if dist_from_closest_food < 1:
            print("Got food!")
            self.reward += 50
            self.env_mesh.respawn_food(closest_food_idx)
            self.total_dist_from_food, self.dist_from_closest_food, self.closest_food_idx = self.calc_dist_from_food()

        angle_to_food = float('inf')
        jelly_vel = self.jellyfish.mesh.world_vel()
        to_foods = self.get_food_dirs()
        if sum(jelly_vel) > 0:
            for to_food in to_foods:
                angle_to_food = min(angle_to_food, np.dot(jelly_vel, to_food) / (np.linalg.norm(jelly_vel) * np.linalg.norm(to_food)))
            self.reward += .5 * (1 - abs(angle_to_food))

        # if dist_from_closest_food-.1 > self.dist_from_closest_food and total_dist_from_food-.1 > self.total_dist_from_food:
        #     self.reward -= .5

        # dist_from_predator = self.calc_dist_from_predator()

        # if dist_from_predator+.05 < self.dist_from_predator:
        #     # print("E")
        #     self.reward -= 2 * abs(self.dist_from_predator - dist_from_predator)
        # elif dist_from_predator-.05 > self.dist_from_predator:
        #     # print("F")
        #     self.reward += 2 * abs(self.dist_from_predator - dist_from_predator)

        # if dist_from_predator < .2:
        #     print("Eaten by predator :(")
        #     self.reward -= 20
        #     self.env_mesh.respawn_predator()
        #     self.dist_from_predator = self.calc_dist_from_predator()

        # don't fire when not at rest
        # if np.sum(action) > .3:
        #     self.reward -= 5

        # jelly_screen_coords = self.jellyfish.mesh.screen_pos()
        # print(self.jellyfish.mesh.pos)
        # if jelly_screen_coords[0] < -1 or jelly_screen_coords[0] > 1 or jelly_screen_coords[1] < -1 or jelly_screen_coords[1] > 1:
        #     self.reward -= .5
        # print(jelly_screen_coords)

        # food_screen_coords = self.env_mesh.food.screen_pos()
        # print(food_screen_coords)

        self.reward -= .002
        # print(self.reward)

        self.total_dist_from_food = total_dist_from_food
        self.dist_from_closest_food = dist_from_closest_food
        self.closest_food_idx = closest_food_idx
        # self.dist_from_predator = dist_from_predator

        state = self.get_state()
        done = self.reward < -5
        info = []

        return state, self.reward, done, info

    def calc_dist_from_food(self):
        food_positions = self.env_mesh.food.pos
        dists = np.linalg.norm(food_positions - self.jellyfish.mesh.world_pos(), axis=1)
        total_dist = np.sum(dists)
        min_dist = np.min(dists)
        closest_food_idx = np.argmin(dists)
        return total_dist, min_dist, closest_food_idx

    def calc_dist_from_predator(self):
        predator_pos = self.env_mesh.predator.world_pos()
        dist = np.linalg.norm(predator_pos - self.jellyfish.mesh.world_pos())
        return dist

    def init_render(self):
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        window = glfw.create_window(self.width, self.height, "Jellyfish Capstone", None, None)
        glfw.make_context_current(window)

        # glViewport(0, 0, width, height)

        # jellyfish = model_loader.load()

        # turn on wireframe mode
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glEnable(GL_DEPTH_TEST)

        return window

    def init_transform_matrices(self):
        camera_pos = glm.vec3(0.0, 3.0, 10.0)
        view = glm.lookAt(camera_pos, glm.vec3(0.0, 3.0, 0.0), self.up)
        projection = glm.mat4(glm.perspective(glm.radians(45.0), self.width / self.height, 0.1, 100.0))
        return view, projection

    def init_models(self):
        view, projection = self.init_transform_matrices()
        jellyfish = model_loader.load(view, projection)
        env_mesh = EnvMesh(view, projection)
        return jellyfish, env_mesh

    # def get_view_matrix(self):
    #     camera_pos = glm.vec3(0.0, self.jellyfish.mesh.world_pos()[1], 10.0)
    #     view = glm.lookAt(camera_pos, glm.vec3(0.0, self.jellyfish.mesh.world_pos()[1], 0.0), self.up)
    #     camera_pos = glm.vec3(0.0, 0.0, 10.0)
    #     view = glm.lookAt(camera_pos, glm.vec3(0.0, 0.0, 0.0), self.up)
    #     return view

    def get_obs_dim(self):
        food_size = self.env_mesh.food_count*3
        jelly_size = 3
        predator_size = 3
        return food_size + jelly_size + predator_size

    # def get_typestate(self):
    #     # jelly_typestate = np.zeros(3)
    #     #
    #     # food_typestate = np.zeros(len(self.env_mesh.food.pos) * 3)
    #     # food_typestate.fill(2)
    #     #
    #     # predator_typestate = np.ones(3) * 3
    #     #
    #     # return np.concatenate((jelly_typestate, food_typestate, predator_typestate))
    #
    #     jelly_typestate = 0 if self.jellyfish.at_rest else 1
    #     food_typestate = [2] * self.env_mesh.food_count
    #     predator_typestate = 3
    #     return np.concatenate(([jelly_typestate], food_typestate, [predator_typestate]))

    def get_food_dirs(self):
        to_foods = []
        jelly_pos = self.jellyfish.mesh.world_pos()
        for food_pos in self.env_mesh.food.world_pos():
            to_food = food_pos - jelly_pos
            to_foods.append(to_food)
        return to_foods

    def get_state(self):
        to_foods = self.get_food_dirs()
        # env_pos = self.env_mesh.world_pos()

        # pos = np.concatenate((jelly_pos, env_pos))

        # typestate = self.get_typestate()
        #
        # return np.concatenate((pos, typestate))
        return np.array(to_foods).flatten()

    def render(self):
        self.draw()
        glfw.poll_events()

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.25, 0.4, 0.55, 1.0)

        # self.view = self.get_view_matrix()

        self.jellyfish.draw()
        self.env_mesh.draw()
        glfw.swap_buffers(self.window)

