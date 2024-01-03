import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    #def take_action(self, state, info):
    def take_action(self, state):
        # state = np.concatenate([
        #     state['wait_packetSize'],
        #     state['channel_state'],
        #     state['LSTM_user'],
        #     state['BSmodel']
        # ])
        # TODO: add info into model.
        # info_li = [info['time'], info['waiting_package_sizes'], info['random_nums'], info['user_coming_package_sizes']]
        # state = torch.tensor([state, info_li], dtype=torch.float).to(self.device)
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        print(f'=========state tensor==============')
        print(state)
        print(f'===========state tensor============')
        # embedding_state_and_wait_package_size = 
        # probs = self.actor(embedding_state_and_info)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        #TODO 根据某些条件，过滤动作
        # action_space_index_total = np.arange(1, 5135)  # 注意，这个应该是action的编号
        # if Vars.time == 0:  # 初始化第0和第1秒的数据，假设这两个好秒内，5个用户都没有数据到达
        #     action_space_matrix = action_space_matrix
        #     # user1
        #     action_illegal_for_user1 = np.where(action_space_matrix[:, 6] != -1)[0]
        #     action_space_index_legal_for_user1 = [num for num in action_space_index_total if num not in action_illegal_for_user1]
        #     # user2
        #     action_illegal_for_user2 = np.where(action_space_matrix[:, 7] != -1)[0]
        #     action_space_index_legal_for_user2 = [num for num in action_space_index_legal_for_user1 if num not in action_illegal_for_user2]
        #     # user3
        #     action_illegal_for_user3 = np.where(action_space_matrix[:, 8] != -1)[0]
        #     action_space_index_legal_for_user3 = [num for num in action_space_index_legal_for_user2 if num not in action_illegal_for_user3]
        #     # user4
        #     action_illegal_for_user4 = np.where(action_space_matrix[:, 9] != -1)[0]
        #     action_space_index_legal_for_user4 = [num for num in action_space_index_legal_for_user3 if num not in action_illegal_for_user4]
        #     # user5
        #     action_illegal_for_user5 = np.where(action_space_matrix[:, 10] != -1)[0]
        #     action_space_index_legal_for_user5 = [num for num in action_space_index_legal_for_user4 if num not in action_illegal_for_user5]

        #     action_space_legal = np.array(action_space_index_legal_for_user5)
        #     action_num = np.random.choice(action_space_legal)
        #     action = action_space_matrix[action_num]

        # else:
        #     action_space_legal, check_matrix = functions.action_choice(action_space_index_total, state, Vars.action_space_possibility)
        #     action_num = np.random.choice(action_space_legal)
        #     action = action_space_matrix[action_num]
        action = action_dist.sample()
        # 从过滤后的动作中采样
        # action = torch.choice(action, 1)
        # return action
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # infos = torch.tensor(transition_dict['dones'], ......)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
