import torch

import gym
import ppo
import base_station_env
import rl_utils
import matplotlib.pyplot as plt

# TODO: changed to embedding.
def _compute_state_dim(obs_space, info):
    state_dim = obs_space.shape[0]
    for key, value in obs_space.spaces.items():
        if key == 'waiting_package_sizes':
            state_dim += value.shape[0] 
        elif key == 'random_nums':
            state_dim += value.shape[0]
        elif key == 'user_coming_package_sizes':
            state_dim += value.shape[0]
        elif key == 'connection_chooses':
            state_dim += value.shape[0] * value.shape[1]
        else:
            raise ValueError('Unknown key: {}'.format(key))
    return state_dim

def main():
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env_name = 'MyBaseStationEnv-v0'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = ppo.PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 21)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()


if __name__ == '__main__':
    main()
