"""
Description: 
version: v1.0
Author: HTY
Date: 2022-11-28 19:15:42
"""

import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from utils import rl_utils


class QNet(torch.nn.Module):
    """ 输入状态，输出动作价值 """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update_rate, device):
        self.action_dim = action_dim
        self.q_net = QNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = QNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update_rate = target_update_rate  # 目标网络更新频率
        self.device = device
        self.count = 0  # 记录更新次数

    def take_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()  # 网络输出的不同动作里取价值最高的
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)    # 按行取对应索引的元素，即对应action的Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)    # 下一个状态最大的Q值
        td_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        dqn_loss = torch.mean(F.mse_loss(q_values, td_targets))     # 损失函数为均方误差
        self.optimizer.zero_grad()  # 梯度清零
        dqn_loss.backward()     # 反向传播
        self.optimizer.step()   # 更新参数

        if self.count % self.target_update_rate == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 同步目标网络的参数
        self.count += 1


if __name__ == "__main__":
    learning_rate = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update_rate = 10
    buffer_size = 10000
    minimal_train_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(torch.__version__)
    print(device)

    env_name = "CartPole-v0"
    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)

    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update_rate, device)

    return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_train_size, batch_size)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()
