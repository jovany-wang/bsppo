from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        arrival_data_matrix0 = np.zeros([max(len(user1_data), len(user2_data), len(user3_data), len(user4_data), len(user5_data)), 3])
                arrival_data_matrix1 = np.zeros([max(len(user1_data), len(user2_data), len(user3_data), len(user4_data), len(user5_data)), 3])
                arrival_data_matrix2 = np.zeros([max(len(user1_data), len(user2_data), len(user3_data), len(user4_data), len(user5_data)), 3])
                done_flag = True  # 用于判断是否为合法情况
                time = 0  # 当前时间，每循环一次time+1，（单位：ms）
                wait_packetSize0 = 0  # 初始化三个基站的等待数据
                wait_packetSize1 = 0
                wait_packetSize2 = 0

                BSmodel0 = 0  # 初始化三个BS的 work model
                BSmodel1 = 0
                BSmodel2 = 0

                energy_sum = 0  # 初始化 三个BS的总能耗

                transiotion_time_ratio0 = 0  # 初始化 三个BS的在1ms内用于转化work model所用时间占总1ms的比例：workmodel转换时间/1ms
                transiotion_time_ratio1 = 0
                transiotion_time_ratio2 = 0

                transition_time0 = 0  # 初始化三个BS work model的转换时间，（单位ms）
                transition_time1 = 0
                transition_time2 = 0

                SM_Hold_Time0 = 0  # 初始化每个基站在某个work model的持续时间
                SM_Hold_Time1 = 0
                SM_Hold_Time2 = 0

                LSTM_user1 = 0  # 初始化LSTM
                LSTM_user2 = 0
                LSTM_user3 = 0
                LSTM_user4 = 0
                LSTM_user5 = 0
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, info = env.reset()
                done = False
                while not done:
                    if (Vars.start_trans_no0 == max(len(user1_data), len(user2_data), len(user3_data), len(user4_data),len(user5_data))) and wait_packetSize0 == 0 and wait_packetSize1 == 0 and wait_packetSize2 == 0:
                        done_flag = False
                    elif time >= max(len(user1_data), len(user2_data), len(user3_data), len(user4_data),len(user5_data)) + 800:
                        done_flag = False
                    else:
                        action = agent.take_action(state, info)
                        arrival_data_matrix0, arrival_data_matrix1, arrival_data_matrix2 = function.arrival_data_matrix(time, user1_data, user2_data, user3_data, user4_data, user5_data, action, arrival_data_matrix0, self.arrival_data_matrix1, self.arrival_data_matrix2)
                        wait_packetSize0, wait_packetSize1, wait_packetSize2 = function.wait_packetSize(time, arrival_data_matrix0, arrival_data_matrix1,arrival_data_matrix2, waitpacketSize0, wait_packetSize1, wait_packetSize2)
                        h_15, trans_rate_list_user1, trans_rate_list_user2, trans_rate_list_user3, trans_rate_list_user4, trans_rate_list_user5 = function.trans_rate_15(action)
                        if self.transition_time0 == 0:  # 处于非转换时间
                            transiotion_time_ratio0, transiotion_time_ratio1, transiotion_time_ratio2, energy_sum, arrival_data_matrix0, arrival_data_matrix1, arrival_data_matrix2 = function.BS_power(time, action, BSmodel0, BSmodel1, BSmodel2, trans_rate0, transiotion_time_ratio0, trans_rate1, transiotion_time_ratio1, trans_rate2, transiotion_time_ratio2, wait_packetSize0, wait_packetSize1, wait_packetSize2, arrival_data_matrix0,arrival_data_matrix1, arrival_data_matrix2, energy_sum, transition_time0, transition_time1, transition_time2)
                            BSmodel0, BSmodel1, BSmodel2, energy_sum, transiotion_time_ratio0, transiotion_time_ratio1, transiotion_time_ratio2, transition_time0, transition_time1, transition_time2, mode_time0, mode_time1, mode_time2 = function.transition(BSmodel0, BSmodel1, BSmodel2, action, energy_sum, transiotion_time_ratio0, transiotion_time_ratio1, transiotion_time_ratio2, transition_time0, transition_time1,transition_time2, Vars.mode_time0, Vars.mode_time1, Vars.mode_time2)
                            Vars.mode_time0 = mode_time0
                            Vars.mode_time1 = mode_time1
                            Vars.mode_time2 = mode_time2
                            LSTM_user1 = LSTM.
                            LSTM_user2 = LSTM.
                            LSTM_user3 = LSTM.
                            LSTM_user4 = LSTM.
                            LSTM_user5 = LSTM.
                        observation = [BSmodel0, BSmodel1, BSmodel2, wait_packetSize0, wait_packetSize1, wait_packetSize2, h_15, LSTM_user1, LSTM_user2, LSTM_user3, LSTM_user4, LSTM_user5]
                        # 计算reward： reward = 0.6*latency + 0.2*energy_sum +0.2*(-LSTM_user1+LSTM_user2+LSTM_user3+LSTM_user4+LSTM_user5)/10000)
                        # 计算latency
                        latency_user0 = sum(arrival_data_matrix0[:-1] - arrival_data_matrix0[:, 1]) / max(len(user1_data), len(user2_data), len(user3_data), len(user4_data), len(user5_data))
                        latency_user1 = sum(arrival_data_matrix1[:-1] - arrival_data_matrix1[:, 1]) / max(len(user1_data), len(user2_data), len(user3_data), len(user4_data), len(user5_data))
                        latency_user2 = sum(arrival_data_matrix2[:-1] - arrival_data_matrix2[:, 1]) / max(len(user1_data), len(user2_data), len(user3_data), len(user4_data), len(user5_data))
                        lagency = (latency_user0 + latency_user1 + latency_user2) / 3
                        reward = 0.6 * lagency + 0.2 * energy_sum + 0.2 * ((-LSTM_user1 + LSTM_user2 + LSTM_user3 + LSTM_user4 + LSTM_user5) / 10000)
                        action = agent.take_action(state, info)
                        next_state, reward, done, info = env.step(action)
                        transition_dict['states'].append(state)
                        transition_dict['actions'].append(action)
                        transition_dict['next_states'].append(next_state)
                        transition_dict['rewards'].append(reward)
                        transition_dict['dones'].append(done)
                        transition_dict['info'].append(info)
                        state = next_state
                        elif transition_time0 > 0:#当前处于转换阶段
                            if transition_time0 < 1:
                                transiotion_time_ratio0 = transition_time0
                                transition_time0 = 0
                            else:
                                transition_time0 = transition_time0 - 1
                        elif transition_time1 > 0
                            if transition_time1 < 1:
                                transiotion_time_ratio1 = transition_time1
                                transition_time1 = 0
                            else:
                                transition_time1 = transition_time1 - 1
                        elif transition_time2 > 0
                            if transition_time2 < 1:
                                transiotion_time_ratio2 = transition_time2
                                transition_time2 = 0
                            else:
                                transition_time2 = transition_time2 - 1
                    time = time+1
                                
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
