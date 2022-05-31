import gym
import numpy as np
import glfw
from OpenGL.GL import *
import glm
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
        food_dists = np.linalg.norm(self.get_food_dirs(), axis=1)
        self.total_dist_from_food, self.dist_from_closest_food, _ = self.calc_dist_from_food(food_dists)

    def reset(self):
        self.jellyfish, self.env_mesh = self.init_models()
        food_dists = np.linalg.norm(self.get_food_dirs(), axis=1)
        self.total_dist_from_food, self.dist_from_closest_food, _ = self.calc_dist_from_food(food_dists)
        self.reward = 0
        return self.get_state()

    def step(self, action):
        current_frame = glfw.get_time() / 3.0
        dt = current_frame - self.last_frame

        rhop_idx = np.argmax(action)
        mnn_delay = np.max(action)
        self.jellyfish.update(current_frame, dt, rhop_idx, mnn_delay)
        self.last_frame = current_frame

        jelly_vel = self.jellyfish.mesh.velocity
        jelly_norm = np.linalg.norm(jelly_vel)
        to_foods = self.get_food_dirs()
        food_norms = np.linalg.norm(to_foods, axis=1)
        if jelly_norm > .01:
            food_dirs = np.zeros(to_foods.shape)
            for i, to_food in enumerate(to_foods):
                food_dirs[i] = to_food / food_norms[i]
            food_angles = abs(np.arccos(np.dot(food_dirs, (jelly_vel / jelly_norm))))
            angle_to_food = np.min(food_angles)
            self.reward += .2 * (1 - abs(angle_to_food))

        total_dist_from_food, dist_from_closest_food, closest_food_idx = self.calc_dist_from_food(food_norms)

        self.reward += .05 * (self.total_dist_from_food - total_dist_from_food)

        self.reward += .1 * (self.dist_from_closest_food - dist_from_closest_food)
        if dist_from_closest_food < 4:
            print("Got food!")
            self.reward += 50
            self.env_mesh.respawn_food(closest_food_idx)
            self.total_dist_from_food, self.dist_from_closest_food, _ = self.calc_dist_from_food(food_norms)

        self.reward -= .002

        self.total_dist_from_food = total_dist_from_food
        self.dist_from_closest_food = dist_from_closest_food

        state = to_foods.flatten()
        done = self.reward < -3
        info = []

        return state, self.reward, done, info

    def calc_dist_from_food(self, food_dists):
        total_dist = np.sum(food_dists)
        min_dist = np.min(food_dists)
        closest_food_idx = np.argmin(food_dists)
        return total_dist, min_dist, closest_food_idx

    def init_render(self):
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        window = glfw.create_window(self.width, self.height, "Jellyfish Capstone", None, None)
        glfw.make_context_current(window)

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

    def get_food_dirs(self):
        return self.env_mesh.food.pos - self.jellyfish.mesh.world_pos()

    def get_state(self):
        to_foods = self.get_food_dirs()
        return np.array(to_foods).flatten()

    def render(self):
        self.draw()
        glfw.poll_events()

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.25, 0.4, 0.55, 1.0)

        self.jellyfish.draw()
        self.env_mesh.draw()
        glfw.swap_buffers(self.window)

