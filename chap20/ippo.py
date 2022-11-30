"""
Description: 独立近端策略优化算法  Independent Proximal Policy Optimization, IPPO
version: v1.0
Author: HTY
Date: 2022-11-29 20:28:12
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from utils import rl_utils
from ma_gym.ma_gym.envs.combat import Combat


class PolicyNet(torch.nn.Module):
    """ 输入状态，输出动作的概率分布，即策略 """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


class ValueNet(torch.nn.Module):
    """ 输入状态，输出状态价值 """
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPO:
    """ 截断式PPO """
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lamda, epsilon, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.lamda = lamda
        self.epsilon = epsilon  # 截断范围
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_error = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lamda, td_error.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        log_probs = torch.log(self.actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        actor_loss = torch.mean(-torch.min(surr1, surr2))
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()


if __name__ == "__main__":
    actor_lr = 3e-4
    critic_lr = 1e-3
    num_episodes = 100000
    hidden_dim = 64
    gamma = 0.99
    lamda = 0.97
    epsilon = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    team_size = 2
    grid_size = (15, 15)
    env = Combat(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size)

    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    # 两个智能体共享同一个策略
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lamda, epsilon, gamma, device)

    win_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                transition_dict_1 = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                transition_dict_2 = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                s = env.reset()
                terminal = False
                while not terminal:
                    a_1 = agent.take_action(s[0])
                    a_2 = agent.take_action(s[1])
                    next_s, r, done, info = env.step([a_1, a_2])

                    transition_dict_1['states'].append(s[0])
                    transition_dict_1['actions'].append(a_1)
                    transition_dict_1['next_states'].append(next_s[0])
                    transition_dict_1['rewards'].append(r[0] + 100 if info['win'] else r[0] - 0.1)
                    transition_dict_1['dones'].append(False)

                    transition_dict_2['states'].append(s[1])
                    transition_dict_2['actions'].append(a_2)
                    transition_dict_2['next_states'].append(next_s[1])
                    transition_dict_2['rewards'].append(r[1] + 100 if info['win'] else r[1] - 0.1)
                    transition_dict_2['dones'].append(False)

                    s = next_s
                    terminal = all(done)

                win_list.append(1 if info["win"] else 0)
                agent.update(transition_dict_1)
                agent.update(transition_dict_2)
                if (i_episode + 1) % 100 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(win_list[-100:])
                    })
                pbar.update(1)

    win_array = np.array(win_list)
    # 每100条轨迹取一次平均
    win_array = np.mean(win_array.reshape(-1, 100), axis=1)

    episodes_list = np.arange(win_array.shape[0]) * 100
    plt.plot(episodes_list, win_array)
    plt.xlabel('Episodes')
    plt.ylabel('Win rate')
    plt.title('IPPO on Combat')
    plt.show()
