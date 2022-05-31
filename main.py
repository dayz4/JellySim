from RLppo.test_train import train, test
import sys

from ocean_env import OceanEnv


def main():
    actor_model = ''
    critic_model = ''
    actor_model = 'ppo_actor.pth'
    critic_model = 'ppo_critic.pth'

    hyperparameters = {
        'timesteps_per_batch': 3000,
        'max_timesteps_per_episode': 500,
        'gamma': 0.95,
        'n_updates_per_iteration': 10,
        'lr': 3e-3,
        'clip': 0.2,
        'render': True,
        'render_every_i': 5
    }

    env = OceanEnv()
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        train(env=env, hyperparameters=hyperparameters, actor_model=actor_model, critic_model=critic_model)
    else:
        test(env=env, actor_model=actor_model)


if __name__ == '__main__':
    main()

