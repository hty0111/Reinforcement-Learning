"""
Description: 信任区域策略优化  trust region policy optimization, TRPO
version: v1.0
Author: HTY
Date: 2022-11-29 16:37:00
"""

import gym
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import sys
sys.path.append("..")
from utils import rl_utils


class PolicyNet(torch.nn.Module):
    """ 输入状态，输出动作的概率分布，即策略 """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    """ 输入状态，输出状态价值 """
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TRPO:
    """ 离散动作 """
    def __init__(self, hidden_dim, state_space, action_space, lamda, kl_constraint, alpha, critic_lr, gamma, device):
        state_dim = state_space.shape[0]
        action_dim = action_space.n
        # 策略网络不需要优化器
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 价值网络需要优化器
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.lamda = lamda  # GAE广义优势估计的超参数
        self.kl_constraint = kl_constraint  # KL库尔贝克-莱布勒距离的最大限制
        self.alpha = alpha  # 线性搜索长度的超参数
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def hessian_vector_product(self, states, old_action_dists, vector):
        """ 计算Hessian矩阵和向量的乘积 """
        # 新策略
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        # 新旧策略之间的平均KL距离（KL散度），即Hessian矩阵
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # KL距离的梯度先和向量点积
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        """ 共轭梯度法求解方程 """
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        r_dot_r = torch.dot(r, r)
        for i in range(10):
            Hp = self.hessian_vector_product(states, old_action_dists, p)
            alpha = r_dot_r / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_r_dot_r = torch.dot(r, r)
            if new_r_dot_r < 1e-10:
                break
            beta = new_r_dot_r / r_dot_r
            p = r + beta * p
            r_dot_r = new_r_dot_r
        return x

    def surrogate_objective(self, states, actions, advantage, old_log_probs, actor):
        """ 计算策略目标 """
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def linear_search(self, states, actions, advantage, old_log_probs, old_action_dists, max_vec):
        """ 线性搜索 """
        old_param = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.surrogate_objective(states, actions, advantage, old_log_probs, self.actor)

        for i in range(15):
            coef = self.alpha**i
            new_param = old_param + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_param, new_actor.parameters())
            new_action_dists = torch.distributions.Categorical(new_actor(states))
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
            new_obj = self.surrogate_objective(states, actions, advantage, old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_param

        return old_param

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):
        """ 更新策略函数 """
        surrogate_obj = self.surrogate_objective(states, actions, advantage, old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # 共轭梯度法求参数更新方向，x = H^(-1)g
        desent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists)
        # 计算Hx向量，避免存储H矩阵
        Hd = self.hessian_vector_product(states, old_action_dists, desent_direction)
        # 根号下 2 * delta / (xT H x)
        max_coef = torch.sqrt(2 * self.kl_constraint / (torch.dot(desent_direction, Hd) + 1e-8))
        # 线性搜索出可行解
        new_param = self.linear_search(states, actions, advantage, old_log_probs, old_action_dists, desent_direction * max_coef)
        # 用线性搜索后的参数更新策略
        torch.nn.utils.convert_parameters.vector_to_parameters(new_param, self.actor.parameters())

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
        old_action_dists = torch.distributions.Categorical(self.actor(states).detach()) # 旧策略
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()    # 更新价值函数
        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)  # 更新策略函数


if __name__ == "__main__":
    """ 离散动作，车杆环境 """
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    critic_lr = 1e-2
    kl_constraint = 0.0005
    alpha = 0.5
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Training on {device}")

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    agent = TRPO(hidden_dim, env.observation_space, env.action_space, lmbda, kl_constraint, alpha, critic_lr, gamma, device)
    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO discrete on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO discrete on {}'.format(env_name))
    plt.show()


