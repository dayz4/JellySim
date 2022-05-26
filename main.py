import glm
from OpenGL.GL import *
from RLppo.test_train import train, test
import gym

import glfw
from ocean_env import OceanEnv

import model_loader
from shader import Shader

width, height = 1600, 1600


# def refresh2d():
#     glViewport(0, 0, width, height)
#     glMatrixMode(GL_PROJECTION)
#     glLoadIdentity()
#     # glOrtho(0.0, width, 0.0, height, 0.0, 1.0)
#     # glMatrixMode (GL_MODELVIEW)
#     glLoadIdentity()


def draw(window, shader_program, jellyfish, t, dt):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(0.2, 0.3, 0.3, 1.0)
    jellyfish.draw(shader_program, t, dt)
    glfw.swap_buffers(window)


# def process_input(window, dt):
#     if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
#         glfw.set_window_should_close(window, True)
#
    # camera_speed = 0.05 * dt
    # if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
    #     pitch += camera_speed
    # if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    #     cameraPos -= cameraSpeed * cameraFront;
    # if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    #     cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) *cameraSpeed;
    # if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    #     cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) *cameraSpeed;


# def configure_camera(shader_program):
#     view = glm.look_at(
#         glm.vec3(0.0, 0.0, 3.0),
#         glm.vec3(0.0, 0,0, 0.0),
#         glm.vec3(0.0, 1.0, 0.0))
#     shader_program.set_matrix("view", view)


def main():
    # glfw.init()
    # glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    # glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    # glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    #
    # window = glfw.create_window(width, height, "Jellyfish Capstone", None, None)
    # glfw.make_context_current(window)
    #
    # # glViewport(0, 0, width, height)
    #
    # shader_program = Shader("shaders/vertex.glsl", "shaders/frag.glsl")
    # jellyfish = model_loader.load()
    #
    # # turn on wireframe mode
    # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    #
    # glEnable(GL_DEPTH_TEST)
    #
    # last_frame = 0
    # while not glfw.window_should_close(window):
    #     current_frame = glfw.get_time() / 2.0
    #     dt = (current_frame - last_frame)
    #     # process_input(window, dt)
    #     draw(window, shader_program, jellyfish, current_frame, dt)
    #     glfw.poll_events()
    #     last_frame = current_frame
    #
    # glfw.terminate()
    # return 0

    # env = OceanEnv()
    # while True:
    #     env.step([0, 0, 0, 1.8, 0, 0, 0, 0])
    #     env.render()

    mode = 'train'
    actor_model = ''
    critic_model = ''
    actor_model = 'ppo_actor.pth'
    critic_model = 'ppo_critic.pth'

    hyperparameters = {
        'timesteps_per_batch': 5000,
        'max_timesteps_per_episode': 500,
        'gamma': 0.9,
        'n_updates_per_iteration': 10,
        'lr': 3e-4,
        'clip': 0.2,
        'render': True,
        'render_every_i': 5
    }

    env = OceanEnv()
    # env = gym.make('MountainCarContinuous-v0')

    if mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model=actor_model, critic_model=critic_model)
    else:
        test(env=env, actor_model=actor_model)


if __name__ == '__main__':
    main()

