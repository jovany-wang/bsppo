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
        self._state = spaces.Tuple([
            # 0, # time. single value with type int
            spaces.MultiDiscrete([
                [4, 3],
                [4, 3],
                [4, 3],
            ]), # states(work states and cover sizes) of 3 base stations.
            spaces.MultiDiscrete([]),
            # np.zeros(3), # waiting package size per base station
            # np.zeros([3, 5]), # N0-N15
            # np.zeros(5), # LSTM_user1-5
        ])

        self.observation_space = self._state
        self.action_space = spaces.Dict({
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



    def step(self, action):
        # 执行给定的动作，并返回新的观测、奖励、是否终止和其他信息
        # return observation, reward, done, info
        return self.observation_space, 11, False, {}

    def reset(self):
        self.seed()
        return self.observation_space

    def render(self, mode='human'):
        pass

    def close(self):
        pass


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
