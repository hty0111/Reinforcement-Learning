"""
Description: 时序差分
version: v1.0
Author: HTY
Date: 2022-11-27 18:50:02
"""

import numpy as np
from matplotlib import pyplot as plt
import gym
from tqdm import tqdm


class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标

    def step(self, action):  # 外部调用这个函数来改变当前位置
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done, None   # 最后一项为调试项

    def reset(self):  # 回归初始状态,坐标轴原点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x


class Sarsa:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, num_action=4):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_action = num_action
        self.Q_table = np.zeros([nrow * ncol, num_action])  # Q(s, a) table

    def take_action(self, state):
        """ 用epsilon贪婪法选择下一步的操作 """
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.num_action)
        else:
            action = np.argmax(self.Q_table[state])     # 最优动作的索引
        return action

    def best_action(self, state):
        """ 打印策略 """
        q_max = np.max(self.Q_table[state])
        a = np.zeros(self.num_action)
        for i in range(self.num_action):
            if self.Q_table[state, i] == q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error
        # 王树森网课中的公式，与上式中的td_error相差一个负号
        # td_target = r + self.gamma * self.Q_table[s1, a1]
        # td_error = self.Q_table[s0, a0] - td_target
        # self.Q_table[s0, a0] = self.Q_table[s0, a0] - self.alpha * td_error


class QLearning:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, num_action=4):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_action = num_action
        self.Q_table = np.zeros([nrow * ncol, num_action])  # Q(s, a) table

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):  # 用于打印策略
        q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.num_action)]
        for i in range(self.num_action):
            if self.Q_table[state, i] == q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1):
        td_error = r + self.gamma * max(self.Q_table[s1]) - self.Q_table[s0, a0]    # 与sarsa的更新方式不同
        self.Q_table[s0, a0] += self.alpha * td_error
        
        
def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    env = env.unwrapped
    env.render()
    holes = set()   # 初始化为集合
    ends = set()
    for s in env.P:
        for a in env.P[s]:
            for trans in env.P[s][a]:
                if trans[2] == 1.0:     # 获得奖励为1，代表是目标
                    ends.add(trans[1])  # 添加索引，与状态对应
                if trans[3] == True:    # 代表游戏结束
                    holes.add(trans[1])
    holes = holes - ends    # 去除到达终点的情况

    env = CliffWalkingEnv(12, 4)

    # np.random.seed(0)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    # agent = Sarsa(env.ncol, env.nrow, epsilon, alpha, gamma)
    agent = QLearning(env.ncol, env.nrow, epsilon, alpha, gamma)
    num_episodes = 500  # 智能体在环境中运行的序列的数量

    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                state = env.reset()
                action = agent.take_action(state)
                done = False
                while not done:
                    next_state, reward, done, _ = env.step(action)
                    next_action = agent.take_action(next_state)
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    # agent.update(state, action, reward, next_state, next_action)
                    agent.update(state, action, reward, next_state)
                    state = next_state
                    action = next_action
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    # action_meaning = ['^', 'v', '<', '>']
    # print('Sarsa算法最终收敛得到的策略为：')
    # print_agent(agent, env, action_meaning, holes, ends)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Sarsa on {}'.format('Frozen Lake'))
    plt.show()
