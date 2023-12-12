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
        '''
        初始化state_space和information中的元素
        '''
        self.time = 0  # 当前时间，每循环一次time+1，（单位：ms）

        self.BSmodel0 = 0  # 初始化三个BS的 work model
        self.BSmodel1 = 0
        self.BSmodel2 = 0

        self.wait_packetSize0 = 0  # 初始化三个基站的等待数据
        self.wait_packetSize1 = 0
        self.wait_packetSize2 = 0

        self.H_15 = 0 #信道状态

        self.LSTM_user1 = 0  # 初始化LSTM
        self.LSTM_user2 = 0
        self.LSTM_user3 = 0
        self.LSTM_user4 = 0
        self.LSTM_user5 = 0
        '''
        初始化在计算state_space元素、reward的过程中需要用到的变量
        '''
        self.energy_sum = 0  # 初始化 三个BS的总能耗

        self.transition_time_ratio0 = 0  # 初始化 三个BS的在1ms内用于转化work model所用时间占总1ms的比例：workmodel转换时间/1ms
        self.transition_time_ratio1 = 0
        self.transition_time_ratio2 = 0

        self.transition_time0 = 0  # 初始化三个BS work model的转换时间，（单位ms）
        self.transition_time1 = 0
        self.transition_time2 = 0

        self.SM_Hold_Time0 = 0  # 初始化每个基站在某个work model的持续时间
        self.SM_Hold_Time1 = 0
        self.SM_Hold_Time2 = 0

        self.arrival_data_matrix0 = np.zeros([5000, 3])#初始化每个基站的到达数据
        self.arrival_data_matrix1 = np.zeros([5000, 3])
        self.arrival_data_matrix2 = np.zeros([5000, 3])

        self.mode_time0 = 0 #初始化每个BS在每个model的持续时间
        self.mode_time1 = 0
        self.mode_time2 = 0

        '''
        疑问：
        1.这里是否需要将变量都写在spaces.MultiDiscrete的矩阵中
        '''
        self.origin_obs_space = spaces.Tuple([
            spaces.MultiDiscrete([
                [4, 3],
                [4, 3],
                [4, 3],
            ]), # states(work states and cover sizes) of 3 base stations.
        ])
         self.origin_action_space = action_space #从action_space.py文件中引入action_space变量
        # self.origin_action_space = spaces.Dict({
        #     'base_stations_modes': spaces.MultiDiscrete([
        #         [4, 3],
        #         [4, 3],
        #         [4, 3],
        #     ]), # WorkMode, CoverSize
        #     'connection_chooses': spaces.MultiDiscrete(
        #         [[2, 2, 2, 2, 2],
        #          [2, 2, 2, 2, 2],
        #          [2, 2, 2, 2, 2],
        #          ]), # A table that has 3 colums and 5 rows, with each value a binary number.
        # })

        self.observation_space = gym.spaces.utils.flatten_space(self.origin_obs_space)
        self.action_space = gym.spaces.utils.flatten_space(self.origin_action_space)

    def step(self, action):
        # 执行给定的动作，并返回新的观测、奖励、是否终止和其他信息
        # return observation, reward, done, info
        info = {
            'time': 0,
            'waiting_package_sizes': np.zeros(3),
            'random_nums': np.zeros(15),
            'LSTM_predict_user_data': np.zeros(5),
            'BSmodel': np.zeros(3),
        }

        self.arrival_data_matrix0, self.arrival_data_matrix1, self.arrival_data_matrix2 = function.arrival_data_matrix(self.time,self.LSTM_user1,self.LSTM_user2,self.LSTM_user3,self.LSTM_user4,self.LSTM_user5,action,arrival_data_matrix0,arrival_data_matrix1,arrival_data_matrix2)
        self.wait_packetSize0,self.wait_packetSize1, self.wait_packetSize2 = function.wait_packetSize(self.time,self.arrival_data_matrix0,self.arrival_data_matrix1,self.arrival_data_matrix2)
        self.H_15,trans_rate_list_user1, trans_rate_list_user2, trans_rate_list_user3, trans_rate_list_user4, trans_rate_list_user5 = function.trans_rate_15(self.action_space)
        trans_rate_BS0, trans_rate_BS1, trans_rate_BS2 = function.trans_rate_3BS(self.action_space,trans_rate_list_user1,trans_rate_list_user2,trans_rate_list_user3,trans_rate_list_user4,trans_rate_list_user5)
        self.transition_time0, self.transition_time1, self.transition_time2, self.transition_time_ratio0, self.transition_time_ratio1,self.transition_time_ratio2, self.energy_sum, self.arrival_data_matrix0, self.arrival_data_matrix1, self.arrival_data_matrix2 = function.BS_power(self.time, self.action_space, self.BSmodel0, self.BSmodel1, self.BSmodel2, trans_rate_BS0, self.transition_time_ratio0,trans_rate_BS1,self.transition_time_ratio1,trans_rate_BS2,self.transition_time_ratio2,self.wait_packetSize0, self.wait_packetSize1, self.wait_packetSize2, self.arrival_data_matrix0, self.arrival_data_matrix1, self.arrival_data_matrix2, self.energy_sum, self.transition_time0,self.transition_time1,self.transition_time2)
        self.BSmodel0, self.BSmodel1, self.BSmodel2, self.energy_sum, self.transition_time_ratio0, self.transition_time_ratio1, self.transition_time_ratio2, self.transition_time0, self.transition_time1, self.transition_time2, self.mode_time0, self.mode_time1, self.mode_time2 = function.transition(self.BSmodel0, self.BSmodel1, self.BSmodel2,self.action_space, self.energy_sum, self.transition_time_ratio0,self.transition_time_ratio1, self.transition_time_ratio2, self.transition_time0, self.transition_time1,self.transition_time2, self.mode_time0, self.mode_time1, self.mode_time2)

        '''
        在这些状态转换完后，再把info的信息放在这里
        '''
        info['time'] += 1  # 每与环境交互一次，时间就相应的+1
        info['waiting_package_sizes'][0] = self.wait_packetSize0
        info['waiting_package_sizes'][1] = self.wait_packetSize1
        info['waiting_package_sizes'][2] = self.wait_packetSize2
        info['random_nums'] = self.H_15
        info['LSTM_predict_user_data'][0] = predicted_value
        info['LSTM_predict_user_data'][0] = predicted_value
        info['LSTM_predict_user_data'][0] = predicted_value
        info['LSTM_predict_user_data'][0] = predicted_value
        info['LSTM_predict_user_data'][0] = predicted_value
        info['BSmodel'][0] = self.BSmodel0
        info['BSmodel'][1] = self.BSmodel1
        info['BSmodel'][2] = self.BSmodel2
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
