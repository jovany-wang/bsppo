import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces import Dict, Discrete
import numpy as np
import function as function
import LSTM as LSTM
import math


gym.envs.register(
    id="MyBaseStationEnv-v0", # 环境名
    entry_point="base_station_env:BaseStationEnv", #接口
    )

class BaseStationEnv(gym.Env):
    def __init__(self):
        self.time = 0
        self.done_flag = True  # 用于判断是否为合法情况
        self.time = 0  # 当前时间，每循环一次time+1，（单位：ms）
        self.wait_packetSize0 = 0  # 初始化三个基站的等待数据
        self.wait_packetSize1 = 0
        self.wait_packetSize2 = 0

        self.BSmodel0 = 0  # 初始化三个BS的 work model
        self.BSmodel1 = 0
        self.BSmodel2 = 0

        self.energy_sum = 0  # 初始化 三个BS的总能耗

        self.transiotion_time_ratio0 = 0  # 初始化 三个BS的在1ms内用于转化work model所用时间占总1ms的比例：workmodel转换时间/1ms
        self.transiotion_time_ratio1 = 0
        self.transiotion_time_ratio2 = 0

        self.transition_time0 = 0  # 初始化三个BS work model的转换时间，（单位ms）
        self.transition_time1 = 0
        self.transition_time2 = 0

        self.SM_Hold_Time0 = 0  # 初始化每个基站在某个work model的持续时间
        self.SM_Hold_Time1 = 0
        self.SM_Hold_Time2 = 0

        self.LSTM_user1 = 0  # 初始化LSTM
        self.LSTM_user2 = 0
        self.LSTM_user3 = 0
        self.LSTM_user4 = 0
        self.LSTM_user5 = 0


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
        while(done_flag):
            if (Vars.start_trans_no0 == max(len(user1_data),len(user2_data),len(user3_data),len(user4_data),len(user5_data))) and wait_packetSize0 == 0 and wait_packetSize1 == 0 and wait_packetSize2 == 0:
                done_flag = False
            elif time >= max(len(user1_data),len(user2_data),len(user3_data),len(user4_data),len(user5_data)) + 800:
                done_flag = False
            else:
                arrival_data_matrix0, arrival_data_matrix1, arrival_data_matrix2 =function.arrival_data_matrix(self.time,user1_data,user2_data,user3_data,user4_data,user5_data,action,self.arrival_data_matrix0,self.arrival_data_matrix1,self.arrival_data_matrix2)
                wait_packetSize0, wait_packetSize1, wait_packetSize2 = function.wait_packetSize(time,arrival_data_matrix0,arrival_data_matrix1,arrival_data_matrix2,waitpacketSize0,wait_packetSize1,wait_packetSize2)
                h_15, trans_rate_list_user1, trans_rate_list_user2, trans_rate_list_user3, trans_rate_list_user4, trans_rate_list_user5 = function.trans_rate_15(action)
                if self.transition_time0 == 0: #处于非转换时间
                    transiotion_time_ratio0, transiotion_time_ratio1, transiotion_time_ratio2, energy_sum, arrival_data_matrix0, arrival_data_matrix1, arrival_data_matrix2 = function.BS_power(time, action,BSmodel0,BSmodel1,BSmodel2, trans_rate0,transiotion_time_ratio0,trans_rate1,transiotion_time_ratio1,trans_rate2,transiotion_time_ratio2,wait_packetSize0, wait_packetSize1, wait_packetSize2,arrival_data_matrix0,arrival_data_matrix1,arrival_data_matrix2,energy_sum,transition_time0,transition_time1,transition_time2)
                    BSmodel0, BSmodel1, BSmodel2, energy_sum, transiotion_time_ratio0, transiotion_time_ratio1, transiotion_time_ratio2, transition_time0, transition_time1, transition_time2, mode_time0, mode_time1, mode_time2 = function.transition(BSmodel0,BSmodel1,BSmodel2,action,energy_sum,transiotion_time_ratio0,transiotion_time_ratio1,transiotion_time_ratio2,transition_time0,transition_time1,transition_time2,Vars.mode_time0,Vars.mode_time1,Vars.mode_time2)
                    Vars.mode_time0 = mode_time0
                    Vars.mode_time1 = mode_time1
                    Vars.mode_time2 = mode_time2
                    LSTM_user1 = LSTM.
                    LSTM_user2 = LSTM.
                    LSTM_user3 = LSTM.
                    LSTM_user4 = LSTM.
                    LSTM_user5 = LSTM.
                observation = [BSmodel0,BSmodel1,BSmodel2,wait_packetSize0,wait_packetSize1,wait_packetSize2,h_15,LSTM_user1,LSTM_user2,LSTM_user3,LSTM_user4,LSTM_user5]
                #计算reward： reward = 0.6*latency + 0.2*energy_sum +0.2*(-LSTM_user1+LSTM_user2+LSTM_user3+LSTM_user4+LSTM_user5)/10000)
                #计算latency
                latency_user0 = sum(arrival_data_matrix0[:-1]-arrival_data_matrix0[:,1])/max(len(user1_data),len(user2_data),len(user3_data),len(user4_data),len(user5_data))
                latency_user1 = sum(arrival_data_matrix1[:-1] - arrival_data_matrix1[:, 1]) / max(len(user1_data),len(user2_data),len(user3_data),len(user4_data),len(user5_data))
                latency_user2 = sum(arrival_data_matrix2[:-1] - arrival_data_matrix2[:, 1]) / max(len(user1_data),len(user2_data),len(user3_data),len(user4_data),len(user5_data))
                lagency =  (latency_user0+latency_user1+latency_user2)/3
                reward = 0.6*lagency + 0.2*energy_sum + 0.2*((-LSTM_user1+LSTM_user2+LSTM_user3+LSTM_user4+LSTM_user5)/10000)
        return observation, reward
        # return observation, reward, done, info
        info = {
            'time': 0,
            'waiting_package_sizes': np.zeros(3),
            'random_nums': np.zeros(15),
            'user_coming_package_sizes': np.zeros(5),
        }
        return self.observation_space.sample(), self._compute_reward(), False, info

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
        return self.observation_space.sample(), info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _compute_reward(self):
        return 0


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
