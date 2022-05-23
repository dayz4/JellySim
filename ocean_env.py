import gym
import numpy as np
import glfw
from OpenGL.GL import *
from shader import Shader
import model_loader


class OceanEnv(gym.Env):
    def __init__(self):
        super(OceanEnv, self).__init__()

        self.width, self.height = 1600, 1600

        self.observation_shape = (10, 10, 10)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.observation_shape,
            dtype=np.uint8
        )
        self.action_space = gym.spaces.Tuple((
            gym.spaces.Discrete(8),
            gym.spaces.Box(
                low=0,
                high=2000,
                shape=(1,),
                dtype=np.uint8
            )))

        self.last_frame = 0
        self.window, self.shader_program = self.init_render()
        self.jellyfish = model_loader.load()

    def reset(self):
        pass

    def step(self, action):
        current_frame = glfw.get_time() / 2.0
        dt = (current_frame - self.last_frame)
        self.jellyfish.update(current_frame, dt)
        self.last_frame = current_frame

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

    def render(self):
        current_frame = glfw.get_time() / 2.0
        dt = (current_frame - self.last_frame)
        self.draw(self.window, self.shader_program, self.jellyfish, current_frame, dt)
        glfw.poll_events()
        self.last_frame = current_frame

    @staticmethod
    def draw(window, shader_program, jellyfish, t, dt):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.2, 0.3, 0.3, 1.0)
        jellyfish.draw(shader_program, t, dt)
        glfw.swap_buffers(window)

