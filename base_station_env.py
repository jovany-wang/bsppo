import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces import Dict, Discrete
import numpy as np


gym.envs.register(
    id="MyBaseStationEnv-v0", # 环境名
    entry_point="base_station_env:BaseStationEnv", #接口
    )

class BaseStationEnv(gym.Env):
    def __init__(self):
        # State_space:
            # time
            # BS0Model，BS1Model，BS2Model，
            # waitpacketSizes0，waitpacketSizes1，waitpacketSizes2，
            # N1-15
            # LSTM_user1, LSTM_user2, LSTM_user3, LSTM_user4, LSTM_user5）
        self.origin_obs_space = spaces.Tuple([
            spaces.MultiDiscrete([
                [4, 3],
                [4, 3],
                [4, 3],
            ]), # states(work states and cover sizes) of 3 base stations.
        ])
        self.origin_action_space = spaces.Dict({
            'base_stations_modes': spaces.MultiDiscrete([
                [4, 3],
                [4, 3],
                [4, 3],
            ]), # WorkMode, CoverSize
            'connection_chooses': spaces.MultiDiscrete(
                [[2, 2, 2, 2, 2],
                 [2, 2, 2, 2, 2],
                 [2, 2, 2, 2, 2],
                 ]), # A table that has 3 colums and 5 rows, with each value a binary number.
        })

        self.observation_space = gym.spaces.utils.flatten_space(self.origin_obs_space)
        self.action_space = gym.spaces.utils.flatten_space(self.origin_action_space)

    def step(self, action):
        # 执行给定的动作，并返回新的观测、奖励、是否终止和其他信息
        # return observation, reward, done, info
        info = self.get_zero_info()
        return self.observation_space.sample(), self._compute_reward(), False, info

    def get_zero_info(self):
        info = {
            'time': 0,
            'waiting_package_sizes': np.zeros(3),
            'random_nums': np.zeros(15),
            'user_coming_package_sizes': np.zeros(5),
        }
        return info

    def reset(self):
        self.seed()
        info = {
            'time': 0,
            # The waiting package sizes for 3 base stations.
            'waiting_package_sizes': np.zeros(3),
            'random_nums': np.zeros(15),
            # The user coming package sizes for 5 users.
            'user_coming_package_sizes': np.zeros(5),
        }
        return self.observation_space.sample(), self.get_zero_info()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _compute_reward(self):
        return 0

    def get_state_dim(self):
        state_dim = self.observation_space.shape[0]
        zero_info = self.get_zero_info()
        state_dim += 1 # time
        state_dim += 3 # waiting package sizes for 3 base stations.
        state_dim += 15 # 15 random numbers.
        return state_dim

    def get_action_dim(self):
        return self.action_space.shape[0]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

def main():
    env = gym.make('MyBaseStationEnv-v0')
    observation = env.reset()
    for _ in range(10):
        action = env.action_space.sample()  # 随机选择一个动作
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            observation = env.reset()

if __name__ == '__main__':
    main()
