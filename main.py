import numpy as np
import torch

import gym

import Vars
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
    '''
    我添加的变量
    1.导入user1-user5的数据集:User1_data,User2_data,User3_data,User4_data,User5_data
    2.用于存储用户数据的matrix，shape为：len(user_data)*3, 3列分别表示 user_data,user_data到达时间，user_data处理完成时间
        arrival_data_matrix0,arrival_data_matrix1,arrival_data_matrix2
     '''
    # 使用NumPy的genfromtxt函数读取CSV文件
    #user1
    User1_data = 'user1_LSTMDate.csv'
    user1_data = np.genfromtxt(User1_data, delimiter=',')
    #user2
    User2_data = 'user1_LSTMDate.csv'
    user2_data = np.genfromtxt(User2_data, delimiter=',')
    #user3
    User3_data = 'user1_LSTMDate.csv'
    user3_data = np.genfromtxt(User3_data, delimiter=',')
    #user4
    User4_data = 'user1_LSTMDate.csv'
    user4_data = np.genfromtxt(User4_data, delimiter=',')
    #user5
    User5_data = 'user1_LSTMDate.csv'
    user5_data = np.genfromtxt(User5_data, delimiter=',')
    arrival_data_matrix0 = np.zeros([max(len(user1_data), len(user2_data), len(user3_data), len(user4_data), len(user5_data)),3])
    arrival_data_matrix1 = np.zeros([max(len(user1_data), len(user2_data), len(user3_data), len(user4_data), len(user5_data)), 3])
    arrival_data_matrix2 = np.zeros([max(len(user1_data), len(user2_data), len(user3_data), len(user4_data), len(user5_data)), 3])


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
    '''
    循环：
    '''
    for episode in range(1000):
        #这些是写在环境中，还是写在main.py函数中
        done_flag = True #用于判断是否为合法情况
        time = 0#当前时间，每循环一次time+1，（单位：ms）
        wait_packetSize0 = 0#初始化三个基站的等待数据
        wait_packetSize1 = 0
        wait_packetSize2 = 0

        BSmodel0 = 0#初始化三个BS的 work model
        BSmodel1 = 0
        BSmodel2 = 0

        energy_sum = 0#初始化 三个BS的总能耗

        transiotion_time_ratio0 = 0#初始化 三个BS的在1ms内用于转化work model所用时间占总1ms的比例：workmodel转换时间/1ms
        transiotion_time_ratio1 = 0
        transiotion_time_ratio2 = 0

        transition_time0 = 0#初始化三个BS work model的转换时间，（单位ms）
        transition_time1 = 0
        transition_time2 = 0

        SM_Hold_Time0 = 0#初始化每个基站在某个work model的持续时间
        SM_Hold_Time1 = 0
        SM_Hold_Time2 = 0

        LSTM_user1= 0#初始化LSTM
        LSTM_user2 = 0
        LSTM_user3 = 0
        LSTM_user4 = 0
        LSTM_user5 = 0


        while(done_flag):
            if (Vars.start_trans_no0 == max(len(user1_data),len(user2_data),len(user3_data),len(user4_data),len(user5_data))) and wait_packetSize0 == 0 and wait_packetSize1 == 0 and wait_packetSize2 == 0:
                done_flag = False
            elif time >= max(len(user1_data),len(user2_data),len(user3_data),len(user4_data),len(user5_data)) + 800:
                done_flag = False
            else:





if __name__ == '__main__':
    main()
