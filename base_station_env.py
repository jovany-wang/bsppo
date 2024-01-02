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
        # # State_space:
        #     # time
        #     # BS0Model，BS1Model，BS2Model，
        #     # waitpacketSizes0，waitpacketSizes1，waitpacketSizes2，
        #     # N1-15
        #     # LSTM_user1, LSTM_user2, LSTM_user3, LSTM_user4, LSTM_user5）
        # self.origin_obs_space = spaces.Tuple([
        #     spaces.MultiDiscrete([
        #         [4, 3],
        #         [4, 3],
        #         [4, 3],
        #     ]), # states(work states and cover sizes) of 3 base stations.
        # ])
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

        # self.observation_space = gym.spaces.utils.flatten_space(self.origin_obs_space)
        # self.action_space = gym.spaces.utils.flatten_space(self.origin_action_space)
        '''
        1. 创建action_space
        '''
        self.action_space = gym.spaces.Discrete(5036)
        '''
        2. 创建state_space
        '''
        self.wait_packetSize_space = spaces.Box(low=0, high=150000, shape=(15,), dtype=np.float32)
        self.channel_state_space = spaces.Box(low=0, high=99, shape=(15,), dtype=np.float32)
        self.LSTM_user_space = spaces.MultiDiscrete([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        self.BSmodel_space = spaces.MultiDiscrete([4, 4, 4, 4])

        # 定义完整的状态空间
        self.observation_space = spaces.Dict({
            'wait_packetSize': self.wait_packetSize_space,
            'channel_state': self.channel_state_space,
            'LSTM_user': self.LSTM_user_space,
            'BSmodel': self.BSmodel_space
        })
        '''
        3.定义state转换过程中的其他中间变量
        '''
        #1.LSTM训练model
        self.scaler1 = MinMaxScaler(feature_range=(0, 1))
        self.scaler2 = MinMaxScaler(feature_range=(0, 1))
        self.scaler3 = MinMaxScaler(feature_range=(0, 1))
        self.scaler4 = MinMaxScaler(feature_range=(0, 1))
        self.scaler5 = MinMaxScaler(feature_range=(0, 1))
        # 时间间隔 LSTM Train
        self.LSTM_model_user1, self.scale1 = functions.LSTM_Train('./user_data/small_interval_LSTM1.csv', self.scaler1)
        self.LSTM_model_user2, self.scale2 = functions.LSTM_Train('./user_data/small_interval_LSTM1.csv', self.scaler2)
        self.LSTM_model_user3, self.scale3 = functions.LSTM_Train('./user_data/small_interval_LSTM1.csv', self.scaler3)
        self.LSTM_model_user4, self.scale4 = functions.LSTM_Train('./user_data/small_interval_LSTM1.csv', self.scaler4)
        self.LSTM_model_user5, self.scale5 = functions.LSTM_Train('./user_data/small_interval_LSTM1.csv', self.scaler5)

        self.LSTM_predict_user1 = pd.read_csv('./user_data/small_interval_LSTM2.csv',header=None).to_numpy()
        self.LSTM_predict_user2 = pd.read_csv('./user_data/small_interval_LSTM2.csv',header=None).to_numpy()
        self.LSTM_predict_user3 = pd.read_csv('./user_data/small_interval_LSTM2.csv',header=None).to_numpy()
        self.LSTM_predict_user4 = pd.read_csv('./user_data/small_interval_LSTM2.csv',header=None).to_numpy()
        self.LSTM_predict_user5 = pd.read_csv('./user_data/small_interval_LSTM2.csv',header=None).to_numpy()
        #2.定义5个user的数据
        self.user1_data = np.zeros([6001, 1])
        self.user2_data = np.zeros([6001, 1])
        self.user3_data = np.zeros([6001, 1])
        self.user4_data = np.zeros([6001, 1])
        self.user5_data = np.zeros([6001, 1])
        #3.其他很多变量
        Vars.time = 0
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

        self.arrival_data_matrix0 = np.zeros([5000, 8])  # 初始化每个基站的到达数据arrival_total,user1,user2,user3,user4,user5,arrival_time, process_finish_time
        self.arrival_data_matrix1 = np.zeros([5000, 8])
        self.arrival_data_matrix2 = np.zeros([5000, 8])

        self.mode_time0 = 0  # 初始化每个BS在每个model的持续时间
        self.mode_time1 = 0
        self.mode_time2 = 0

        self.done = True  # 完成一个epoch的标记，这个初始化不用特意给他初始化了，已经在state_space中给他初始化了，就不用管了

        Vars.counter_1_for_previous_5 = 0
        Vars.counter_2_for_previous_5 = 0
        Vars.counter_3_for_previous_5 = 0
        Vars.counter_4_for_previous_5 = 0
        Vars.counter_5_for_previous_5 = 0
        Vars.counter_1_for_LSTM = 0
        Vars.counter_2_for_LSTM = 0
        Vars.counter_3_for_LSTM = 0
        Vars.counter_4_for_LSTM = 0
        Vars.counter_5_for_LSTM = 0
        #初始化延迟
        self.delay_BS0 = 0
        self.delay_BS1 = 0
        self.delay_BS2 = 0
        self.delay = 0
        #5个user对应的3个BS的传输
        self.trans_rate_list_user1 = []
        self.trans_rate_list_user2 = []
        self.trans_rate_list_user3 = []
        self.trans_rate_list_user4 = []
        self.trans_rate_list_user5 = []

    def step(self, action):
        # 执行给定的动作，并返回新的观测、奖励、是否终止和其他信息
        # return observation, reward, done, info
        #info = self.get_zero_info()
        #return self.observation_space.sample(), self._compute_reward(), False, info
        #将action对应action_matrix
        action_row = Vars.action_space_possibility[action]
        # 获取当前的state
        self.arrival_data_matrix0, self.arrival_data_matrix1, self.arrival_data_matrix2 = functions.arrival_data_matrix(self.user1_data, self.user2_data, self.user3_data, self.user4_data, self.user5_data, action_row, self.arrival_data_matrix0, self.arrival_data_matrix1, self.arrival_data_matrix2)
        wait_packetSize = functions.wait_packetSize(self.arrival_data_matrix0, self.arrival_data_matrix1, self.arrival_data_matrix2,self.observation)
        H_15,self.trans_rate_list_user1, self.trans_rate_list_user2, self.trans_rate_list_user3, self.trans_rate_list_user4, self.trans_rate_list_user5 = functions.trans_rate_15(action_row)
        self.transition_time0, self.transition_time1, self.transition_time2, self.transition_time_ratio0, self.transition_time_ratio1, self.transition_time_ratio2, self.energy_sum, self.arrival_data_matrix0, self.arrival_data_matrix1, self.arrival_data_matrix2 = functions.BS_power(action_row,self.observation,self.transition_time_ratio0, self.transition_time_ratio1,self.transition_time_ratio2, self.arrival_data_matrix0, self.arrival_data_matrix1, self.arrival_data_matrix2, self.energy_sum, self.transition_time0, self.transition_time1, self.transition_time2,self.trans_rate_list_user1, self.trans_rate_list_user2, self.trans_rate_list_user3, self.trans_rate_list_user4, self.trans_rate_list_user5)
        self.energy_sum, self.transition_time_ratio0, self.transition_time_ratio1, self.transition_time_ratio2, self.transition_time0, self.transition_time1, self.transition_time2, self.mode_time0, self.mode_time1, self.mode_time2 = functions.transition(self.observation, action_row, self.energy_sum, self.transition_time_ratio0, self.transition_time_ratio1, self.transition_time_ratio2, self.transition_time0, self.transition_time1, self.transition_time2, self.mode_time0, self.mode_time1, self.mode_time2)
        # LSTM
        if Vars.time <= 71:
            # 当时间小于5时，因为需要用前5个数据做LSTM的预测，因此，只能读取csv中的数据，从而得到下一次用户到达是什么时候
            if Vars.time == 0:
                time_next_arrival_user1 = self.LSTM_predict_user1[Vars.counter_1_for_previous_5]  # 获取时间间隔 narry
                LSTM_user1_arrivalTime = int((Vars.time + time_next_arrival_user1)[0])
                time_next_arrival_user2 = self.LSTM_predict_user2[Vars.counter_2_for_previous_5]
                LSTM_user2_arrivalTime = int((Vars.time + time_next_arrival_user2)[0])
                time_next_arrival_user3 = self.LSTM_predict_user3[Vars.counter_3_for_previous_5]
                LSTM_user3_arrivalTime = int((Vars.time + time_next_arrival_user3)[0])
                time_next_arrival_user4 = self.LSTM_predict_user4[Vars.counter_4_for_previous_5]
                LSTM_user4_arrivalTime = int((Vars.time + time_next_arrival_user4)[0])
                time_next_arrival_user5 = self.LSTM_predict_user5[Vars.counter_5_for_previous_5]
                LSTM_user5_arrivalTime = int((Vars.time + time_next_arrival_user5)[0])

            # 当前时间 0<time<5
            else:

                if Vars.time < LSTM_user1_arrivalTime:
                    pass
                else:
                    Vars.counter_1_for_previous_5 += 1
                    # 先给userdata赋个值，再重新计算新的到达时间
                    user1_data[LSTM_user1_arrivalTime][0] = 500000
                    time_next_arrival_user1 = self.LSTM_predict_user1[Vars.counter_1_for_previous_5]  # 获取时间间隔 narry
                    LSTM_user1_arrivalTime = int((Vars.time + time_next_arrival_user1)[0])  # 时间间隔+当前时间 = 下次数据到达时刻 narry
                    # 根据到达时间，给userdata赋值

                if Vars.time < LSTM_user2_arrivalTime:
                    pass
                else:
                    Vars.counter_2_for_previous_5 += 1
                    user2_data[LSTM_user2_arrivalTime][0] = 500000
                    time_next_arrival_user2 = self.LSTM_predict_user2[Vars.counter_2_for_previous_5]
                    LSTM_user2_arrivalTime = int((Vars.time + time_next_arrival_user2)[0])  # 将到达时间转换为int类型，而不是array型
                if Vars.time < LSTM_user3_arrivalTime:
                    pass
                else:
                    Vars.counter_3_for_previous_5 += 1
                    user3_data[LSTM_user3_arrivalTime][0] = 500000
                    time_next_arrival_user3 = self.LSTM_predict_user3[Vars.counter_3_for_previous_5]
                    LSTM_user3_arrivalTime = int((Vars.time + time_next_arrival_user3)[0])
                if Vars.time < LSTM_user4_arrivalTime:
                    pass
                else:
                    Vars.counter_4_for_previous_5 += 1
                    user4_data[LSTM_user4_arrivalTime][0] = 500000
                    time_next_arrival_user4 = self.LSTM_predict_user4[Vars.counter_4_for_previous_5]
                    LSTM_user4_arrivalTime = int((Vars.time + time_next_arrival_user4)[0])
                if Vars.time < LSTM_user5_arrivalTime:
                    pass
                else:
                    Vars.counter_5_for_previous_5 += 1
                    user5_data[LSTM_user5_arrivalTime][0] = 500000
                    time_next_arrival_user5 = self.LSTM_predict_user5[Vars.counter_5_for_previous_5]
                    LSTM_user5_arrivalTime = int((Vars.time + time_next_arrival_user5)[0])

        else:  # 当时间大于前5个时间间隔的总和时，用LSTM的预测预测数据
            # for user1
            if Vars.time < LSTM_user1_arrivalTime:  # 说明当前的时间还在预测范围内，则这次就可以不用预测，直到下一次数据到达时，再预测
                pass
            else:  # 说明下次的数据已经到达，需要再做一次预测了
                time_next_arrival_user1 = functions.LSTM_predict(
                    self.LSTM_model_user1,
                    './user_data/small_interval_LSTM2.csv',
                    self.scaler1,
                    Vars.counter_1_for_LSTM)  # 时间间隔
                LSTM_user1_arrivalTime = int((Vars.time + time_next_arrival_user1))  # 下一次数据到达时间 = 当前时间+ 时间间隔
                user1_data[LSTM_user1_arrivalTime][0] = 500000
                Vars.counter_1_for_LSTM += 1

            # for user2
            if Vars.time < LSTM_user2_arrivalTime:
                pass
            else:
                time_next_arrival_user2 = functions.LSTM_predict(
                    self.LSTM_model_user2,
                    './user_data/small_interval_LSTM2.csv',
                    self.scaler2,
                    Vars.counter_2_for_LSTM)  # 时间间隔
                LSTM_user2_arrivalTime = int((Vars.time + time_next_arrival_user2))  # 下一次数据到达时间 = 当前时间+ 时间间隔
                user2_data[LSTM_user2_arrivalTime][0] = 500000
                Vars.counter_2_for_LSTM += 1

            # for user3
            if Vars.time < LSTM_user3_arrivalTime:
                pass
            else:
                time_next_arrival_user3 = functions.LSTM_predict(
                    self.LSTM_model_user3,
                    './user_data/small_interval_LSTM2.csv',
                    self.scaler3,
                    Vars.counter_3_for_LSTM)  # 时间间隔
                LSTM_user3_arrivalTime = int((Vars.time + time_next_arrival_user3))  # 下一次数据到达时间 = 当前时间+ 时间间隔
                user3_data[LSTM_user3_arrivalTime][0] = 500000
                Vars.counter_3_for_LSTM += 1

            # for user4
            if Vars.time < LSTM_user4_arrivalTime:
                pass
            else:
                time_next_arrival_user4 = functions.LSTM_predict(
                    self.LSTM_model_user4,
                    './user_data/small_interval_LSTM2.csv',
                    self.scaler4,
                    Vars.counter_4_for_LSTM)  # 时间间隔
                LSTM_user4_arrivalTime = int((Vars.time + time_next_arrival_user4))  # 下一次数据到达时间 = 当前时间+ 时间间隔
                user4_data[LSTM_user4_arrivalTime][0] = 500000
                Vars.counter_4_for_LSTM += 1

            # for user5
            if Vars.time < LSTM_user5_arrivalTime:
                pass
            else:
                time_next_arrival_user5 = functions.LSTM_predict(
                    self.LSTM_model_user5,
                    './user_data/small_interval_LSTM2.csv',
                    sself.caler5,
                    Vars.counter_5_for_LSTM)  # 时间间隔
                LSTM_user5_arrivalTime = int((Vars.time + time_next_arrival_user5))  # 下一次数据到达时间 = 当前时间+ 时间间隔
                user5_data[LSTM_user5_arrivalTime][0] = 500000
                Vars.counter_5_for_LSTM += 1
        # 给LSTM赋值，如果当前的时间等于数据到达时间，则LSTM值为1，否则LSTM值为0
        # user1

        if Vars.time == LSTM_user1_arrivalTime - 1:  # 到达时间-1证明当前时刻是下次时间到达的前一个时刻
            LSTM_user_1 = 1  # 1表示有数据到达，0表示没有数据到达，就直接不用到达数据包的大小了，-1表示当前时隙是数据到达之前的哪一个时隙
        else:
            LSTM_user_1 = 0

        # user2
        if Vars.time == LSTM_user2_arrivalTime - 1:
            LSTM_user_2 = 1
        else:
            LSTM_user_2 = 0

        # user3
        if Vars.time == LSTM_user3_arrivalTime - 1:
            LSTM_user_3 = 1
        else:
            LSTM_user_3 = 0

        # user4
        if Vars.time == LSTM_user4_arrivalTime - 1:
            LSTM_user_4 = 1
        else:
            LSTM_user_4 = 0

        # user5
        if Vars.time == LSTM_user5_arrivalTime - 1:
            LSTM_user_5 = 1
        else:
            LSTM_user_5 = 0

        #对state进行更新
        self.observation['wait_packetSize'] = wait_packetSize
        self.observation['channel_state'] = H_15
        self.observation['LSTM_user'][0] = self.observation['LSTM_user'][5]
        self.observation['LSTM_user'][1] = self.observation['LSTM_user'][6]
        self.observation['LSTM_user'][2] = self.observation['LSTM_user'][7]
        self.observation['LSTM_user'][3] = self.observation['LSTM_user'][8]
        self.observation['LSTM_user'][4] = self.observation['LSTM_user'][9]
        self.observation['LSTM_user'][5] = LSTM_user_1
        self.observation['LSTM_user'][6] = LSTM_user_2
        self.observation['LSTM_user'][7] = LSTM_user_3
        self.observation['LSTM_user'][8] = LSTM_user_4
        self.observation['LSTM_user'][9] = LSTM_user_5
        self.observation['BSmodel'][0] = action_row[0]
        self.observation['BSmodel'][1] = action_row[1]
        self.observation['BSmodel'][2] = action_row[2]
        #done
        if Vars.time > 3000:
            done = False
        elif Vars.time <= 3000:
            done = True
        #reward
        if np.sum(wait_packetSize) >0:
            reward = -0.7*(np.sum(wait_packetSize)) + 0.3* self.energy_sum
        else:
            reward = self.energy_sum * 10
        info = dict()
        return self.observation,reward,done,info

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
        # info = {
        #     'time': 0,
        #     # The waiting package sizes for 3 base stations.
        #     'waiting_package_sizes': np.zeros(3),
        #     'random_nums': np.zeros(15),
        #     # The user coming package sizes for 5 users.
        #     'user_coming_package_sizes': np.zeros(5),
        # }
        # return self.observation_space.sample(), self.get_zero_info()
        self.observation = {
        'wait_packetSize': np.zeros(15),
        'channel_state': np.zeros(15),
        'LSTM_user': np.zeros(10),
        'BSmodel': np.zeros(4)
        }
        return self.observation

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _compute_reward(self):
        return 0

    def get_state_dim(self):
        # state_dim = self.observation_space.shape[0]
        # zero_info = self.get_zero_info()
        # state_dim += 1 # time
        # state_dim += 3 # waiting package sizes for 3 base stations.
        # state_dim += 15 # 15 random numbers.
        # return state_dim
        pass

    def get_action_dim(self):
        return self.action_space.shape[0]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

def main():
    env = gym.make('MyBaseStationEnv-v0')
    observation = env.reset()
    # for _ in range(10):
    for _ in range(3000):
        action = env.action_space.sample()  # 随机选择一个动作
        observation, reward, done, info = env.step(action)
        env.render()
        Vars.time += 1
        if done:
            observation = env.reset()

if __name__ == '__main__':
    main()
