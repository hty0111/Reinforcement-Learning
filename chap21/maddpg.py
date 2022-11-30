"""
Description: multi-agent deep deterministic policy gradient, MADDPG
version: v1.0
Author: HTY
Date: 2022-11-29 21:38:16
"""

import random
import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from utils import rl_utils
from multiagent_particle_envs.multiagent.environment import MultiAgentEnv
from multiagent_particle_envs.make_env import make_env


def onehot_from_logits(logits, eps=0.01):
    """ 将全连接层的输出转化为one-hot """
    argmax_actions = (logits == logits.max(1, keepdim=True)[0]).float()
    rand_actions = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
        ]], requires_grad=False).to(logits.device)
    # 用过epsilon贪婪选择动作
    return torch.stack([argmax_actions[i] if r > eps else rand_actions[i] for i, r in enumerate(torch.rand(logits.shape[0]))])


def gumbel_sample(shape, eps=1e-20, tensor_type=torch.FloatTensor):
    """ 从gumbel(0, 1)分布中采样 """
    U = torch.autograd.Variable(tensor_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ 从gumbel-softmax中采样 """
    y = logits + gumbel_sample(logits.shape, tensor_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """ 采样并离散化 """
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量，其梯度是y，因此这是一个能与环境交互的离散动作，同时可以计算反向梯度
    return y


class FC(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FC, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, critic_input_dim, actor_lr, critic_lr, tau, device):
        self.actor = FC(state_dim, hidden_dim, action_dim).to(device)
        self.critic = FC(critic_input_dim, hidden_dim, 1).to(device)
        self.target_actor = FC(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = FC(critic_input_dim, hidden_dim, 1).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.tau = tau

    def take_action(self, state, explore=False):
        action = self.actor(state)
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


class MADDPG:
    def __init__(self, env, state_dims, hidden_dim, action_dims, critic_input_dim, actor_lr, critic_lr, gamma, tau, device):
        self.agents = []
        self.num_agents = len(env.agents)
        for i in range(self.num_agents):
            self.agents.append(DDPG(state_dims[i], hidden_dim, action_dims[i], critic_input_dim, actor_lr, critic_lr, tau, device))
        self.gamma = gamma
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device

    def policies(self):
        return [agent.actor for agent in self.agents]

    def target_policies(self):
        return [agent.target_actor for agent in self.agents]

    def take_action(self, states, explore):
        states = [torch.tensor([states[i]], dtype=torch.float, device=self.device) for i in range(self.num_agents)]
        return [agent.take_action(state, explore) for agent, state in zip(self.agents, states)]

    def update(self, sample, idx_agent):
        state, action, reward, next_state, done = sample
        current_agent = self.agents[idx_agent]

        current_agent.critic_optimizer.zero_grad()
        all_target_action = [onehot_from_logits(pi(next_s)) for pi, next_s in zip(self.target_policies(), next_state)]
        target_critic_input = torch.cat((*next_state, *all_target_action), dim=1)
        target_critic_value = reward[idx_agent].view(-1, 1) + \
                              self.gamma * current_agent.target_critic(target_critic_input) * (1 - done[idx_agent].view(-1, 1))
        critic_input = torch.cat((*state, *action), dim=1)
        critic_value = current_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())
        critic_loss.backward()
        current_agent.critic_optimizer.step()

        current_agent.actor_optimizer.zero_grad()
        current_actor_out = current_agent.actor(state[idx_agent])
        current_actor_action = gumbel_softmax(current_actor_out)
        all_actor_actions = []
        for i, (pi, s) in enumerate(zip(self.policies(), state)):
            if i == idx_agent:
                all_actor_actions.append(current_actor_action)
            else:
                all_actor_actions.append(onehot_from_logits(pi(s)))

        actor_loss = -current_agent.critic(torch.cat((*state, *all_actor_actions), dim=1)).mean()
        actor_loss += (current_actor_out**2).mean() * 1e-3
        actor_loss.backward()
        current_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agent in self.agents:
            agent.soft_update(agent.actor, agent.target_actor)
            agent.soft_update(agent.critic, agent.target_critic)


def evaluate(env_id, maddpg, n_episode=10, episode_length=25):
    # 对学习的策略进行评估,此时不会进行探索
    env = make_env(env_id)
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        obs = env.reset()
        for t_i in range(episode_length):
            actions = maddpg.take_action(obs, explore=False)
            obs, rew, done, info = env.step(actions)
            rew = np.array(rew)
            returns += rew / n_episode
    return returns.tolist()


if __name__ == "__main__":
    num_episodes = 5000
    episode_length = 25
    buffer_size = 100000
    actor_lr = 1e-2
    critic_lr = 1e-2
    gamma = 0.95
    tau = 1e-2
    batch_size = 1024
    update_interval = 100
    minimal_size = 4000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device {device}")

    env_id = "simple_adversary"
    env = make_env(env_id)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)

    hidden_dim = 64
    state_dims = []
    action_dims = []
    for action_space in env.action_space:
        action_dims.append(action_space.n)
    for state_space in env.observation_space:
        state_dims.append(state_space.shape[0])
    critic_input_dim = sum(state_dims) + sum(action_dims)

    maddpg = MADDPG(env, state_dims, hidden_dim, action_dims, critic_input_dim, actor_lr, critic_lr, gamma, tau, device)

    return_list = []  # 记录每一轮的回报（return）
    total_step = 0
    for i_episode in range(num_episodes):
        state = env.reset()
        # ep_returns = np.zeros(len(env.agents))
        for e_i in range(episode_length):
            actions = maddpg.take_action(state, explore=True)
            next_state, reward, done, _ = env.step(actions)
            replay_buffer.add(state, actions, reward, next_state, done)
            state = next_state

            total_step += 1
            if replay_buffer.size() >= minimal_size and total_step % update_interval == 0:
                sample = replay_buffer.sample(batch_size)

                def stack_array(x):
                    rearranged = [[sub_x[i] for sub_x in x] for i in range(len(x[0]))]
                    return [torch.FloatTensor(np.vstack(aa)).to(device) for aa in rearranged]

                sample = [stack_array(x) for x in sample]
                for a_i in range(len(env.agents)):
                    maddpg.update(sample, a_i)
                maddpg.update_all_targets()
        if (i_episode + 1) % 100 == 0:
            ep_returns = evaluate(env_id, maddpg, n_episode=100)
            return_list.append(ep_returns)
            print(f"Episode: {i_episode + 1}, {ep_returns}")

    return_array = np.array(return_list)
    for i, agent_name in enumerate(["adversary_0", "agent_0", "agent_1"]):
        plt.figure()
        plt.plot(
            np.arange(return_array.shape[0]) * 100,
            rl_utils.moving_average(return_array[:, i], 9))
        plt.xlabel("Episodes")
        plt.ylabel("Returns")
        plt.title(f"{agent_name} by MADDPG")
