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
        self.window, self.shader_program = self.init_render()
        self.view, self.projection = self.init_transform_matrices()

        self.jellyfish = model_loader.load()
        self.env_mesh = self.init_env_mesh()

        obs_dim = self.get_obs_dim()
        pos_bound = np.zeros(obs_dim)
        pos_bound.fill(10)
        element_count = 2 + self.env_mesh.food_count
        self.observation_space = gym.spaces.Box(
            low=np.concatenate((
                -pos_bound,
                -np.ones(obs_dim),
                np.zeros(element_count))).flatten(),
            high=np.concatenate((
                pos_bound,
                np.ones(obs_dim),
                np.ones(element_count)*3)),
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

    def reset(self):
        self.jellyfish = model_loader.load()
        self.env_mesh = EnvMesh()
        self.total_dist_from_food, self.dist_from_closest_food, self.closest_food_idx = self.calc_dist_from_food()
        self.reward = 0
        return self.get_state()

    def step(self, action):
        # print("action", action)
        current_frame = glfw.get_time() / 3.0
        dt = current_frame - self.last_frame
        # print(dt)

        # rhop_idx = action[0]
        # mnn_delay = action[1][rhop_idx]
        # rhop_idx = max(range(len(action)), key=values.__getitem__)
        rhop_idx = np.argmin(action)
        mnn_delay = np.max(action)

        self.jellyfish.update(current_frame, dt, rhop_idx, mnn_delay)
        self.last_frame = current_frame

        total_dist_from_food, dist_from_closest_food, closest_food_idx = self.calc_dist_from_food()

        if total_dist_from_food-.01 > self.total_dist_from_food:
            self.reward -= 1
        elif total_dist_from_food+.01 < self.total_dist_from_food:
            self.reward += 5 * (self.total_dist_from_food - total_dist_from_food)

        if dist_from_closest_food-.01 > self.dist_from_closest_food:
            self.reward -= .5
        elif dist_from_closest_food+.01 < self.dist_from_closest_food:
            self.reward += 2.5 * (self.dist_from_closest_food - dist_from_closest_food)

        if dist_from_closest_food < .5:
            print("Got food!")
            self.reward += 10
            self.env_mesh.respawn_food(closest_food_idx)

        # don't fire when not at rest
        # if np.sum(action) > .3:
        #     self.reward -= 5

        self.reward -= .001

        self.total_dist_from_food = total_dist_from_food
        self.dist_from_closest_food = dist_from_closest_food
        self.closest_food_idx = closest_food_idx

        state = self.get_state()
        done = self.reward < -15
        info = []

        return state, self.reward, done, info

    def calc_dist_from_food(self):
        food_positions = self.env_mesh.food.pos
        dists = np.linalg.norm(food_positions - self.jellyfish.mesh.world_pos_coord(), axis=1)
        total_dist = np.sum(dists)
        min_dist = np.min(dists)
        closest_food_idx = np.argmin(dists)
        return total_dist, min_dist, closest_food_idx

    def init_render(self):
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        window = glfw.create_window(self.width, self.height, "Jellyfish Capstone", None, None)
        glfw.make_context_current(window)

        # glViewport(0, 0, width, height)

        shader_program = Shader("shaders/vertex.glsl", "shaders/frag.glsl")
        # jellyfish = model_loader.load()

        # turn on wireframe mode
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glEnable(GL_DEPTH_TEST)

        return window, shader_program

    def init_env_mesh(self):
        return EnvMesh()

    def init_transform_matrices(self):
        # camera_pos = glm.vec3(0.0, 3.0, 10.0)
        camera_pos = glm.vec3(0.0, 0.0, 10.0)
        view = glm.lookAt(camera_pos, glm.vec3(0.0, 0.0, 0.0), self.up)
        projection = glm.mat4(glm.perspective(glm.radians(45.0), 1600 / 1600, 0.1, 100.0))
        return view, projection

    def get_obs_dim(self):
        food_size = self.env_mesh.food_count*3
        jelly_size = 3
        predator_size = 3
        return food_size + jelly_size + predator_size

    def get_typestate(self):
        # jelly_typestate = np.zeros(3)
        #
        # food_typestate = np.zeros(len(self.env_mesh.food.pos) * 3)
        # food_typestate.fill(2)
        #
        # predator_typestate = np.ones(3) * 3
        #
        # return np.concatenate((jelly_typestate, food_typestate, predator_typestate))

        jelly_typestate = 0 if self.jellyfish.at_rest else 1
        food_typestate = [2] * self.env_mesh.food_count
        predator_typestate = 3
        return np.concatenate(([jelly_typestate], food_typestate, [predator_typestate]))

    def get_state(self):
        jelly_pos, jelly_vel = self.jellyfish.mesh.world_pos_velocity()
        env_pos, env_vel = self.env_mesh.world_pos_velocity()

        pos = np.concatenate((jelly_pos, env_pos))
        vel = np.concatenate((jelly_vel, env_vel))

        typestate = self.get_typestate()

        return np.concatenate((pos, vel, typestate))

    def render(self):
        current_frame = glfw.get_time() / 2.0
        dt = (current_frame - self.last_frame)

        self.draw(current_frame, dt)
        glfw.poll_events()
        # self.last_frame = current_frame

    def draw(self, t, dt):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.25, 0.4, 0.55, 1.0)
        self.jellyfish.draw(self.shader_program, self.view, self.projection, t, dt)
        self.env_mesh.draw(self.view, self.projection)
        glfw.swap_buffers(self.window)

