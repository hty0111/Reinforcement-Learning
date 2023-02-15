"""
Description: 动态规划：策略迭代 & 价值迭代
version: v1.0
Author: HTY
Date: 2022-11-27 15:21:17
"""

import copy
import gym


class PolicyIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.state_num = self.env.ncol * self.env.nrow  # 状态个数
        self.V = [0] * self.state_num   # 初始化每个状态的价值为0
        self.pi = [[0.25, 0.25, 0.25, 0.25] for i in range(self.state_num)]     # 初始化每个状态的策略为均匀随机，每个状态s采取动作a的概率分布

    def policy_evaluation(self):
        cnt = 1     # 计数器
        while True:
            max_diff = 0
            new_V = [0] * self.state_num
            for s in range(self.state_num):
                Q_list = []  # 当前状态下所有动作对应的Q(s, a)
                for a in range(4):
                    Q = 0
                    # 遍历s'，trans为转移到不同状态s_next的概率p和奖励r
                    for trans in self.env.P[s][a]:
                        p, s_next, r, done = trans
                        # 对不同的状态进行累加，根据公式更新，这里的奖励与下一个状态有关
                        Q += p * r + self.gamma * p * self.V[s_next] * (1 - done)
                    Q_list.append(self.pi[s][a] * Q)
                new_V[s] = sum(Q_list)   # Q对a求和得到当前状态的V
                max_diff = max(max_diff, abs(new_V[s] - self.V[s])) # V^{k+1} - V^{k}
            self.V = new_V
            if max_diff < self.theta:   # 两次状态价值函数的差小于阈值时停止迭代
                break
            cnt += 1
        print(f"Finished policy evaluation after {cnt} rounds.")

    def policy_improvement(self):
        for s in range(self.state_num):
            Q_list = []
            for a in range(4):
                Q = 0
                for trans in self.env.P[s][a]:
                    p, s_next, r, done = trans
                    Q += p * r + self.gamma * p * self.V[s_next] * (1 - done)
                Q_list.append(Q)
            max_Q = max(Q_list)
            cnt_max_Q = Q_list.count(max_Q)  # 计算有几个动作得到了最大的Q
            self.pi[s] = [1 / cnt_max_Q if Q == max_Q else 0 for Q in Q_list]    # 提升策略
        print("Finished policy improvement")
        return self.pi

    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()
            if new_pi == old_pi:
                break


class ValueIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.state_num = self.env.ncol * self.env.nrow  # 状态个数
        self.V = [0] * self.state_num   # 初始化每个状态的价值为0
        self.pi = [None for i in range(self.state_num)]     # 价值迭代结束后得到的策略

    def value_iteration(self):
        cnt = 1     # 计数器
        while True:
            max_diff = 0
            new_V = [0] * self.state_num
            for s in range(self.state_num):
                Q_list = []
                for a in range(4):
                    Q = 0
                    for trans in self.env.P[s][a]:
                        p, s_next, r, done = trans
                        Q += p * (r + self.gamma * self.V[s_next] * (1 - done))
                    Q_list.append(Q)
                new_V[s] = max(Q_list)   # 选择最大的Q作为新的V，进行一步策略更新
                max_diff = max(max_diff, abs(new_V[s] - self.V[s]))
            self.V = new_V
            if max_diff < self.theta:
                break
            cnt += 1
        print(f"Finished value evaluation after {cnt} rounds.")
        self.get_policy()

    def get_policy(self):
        for s in range(self.state_num):
            Q_list = []
            for a in range(4):
                Q = 0
                for trans in self.env.P[s][a]:
                    p, s_next, r, done = trans
                    Q += p * r + self.gamma * p * self.V[s_next] * (1 - done)
                Q_list.append(Q)
            max_Q = max(Q_list)
            cnt_max_Q = Q_list.count(max_Q)  # 计算有几个动作得到了最大的Q
            self.pi[s] = [1 / cnt_max_Q if Q == max_Q else 0 for Q in Q_list]    # 获得策略


def print_agent(agent, action_meaning, disaster=None, end=None):
    if end is None:
        end = []
    if disaster is None:
        disaster = []
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.V[i * agent.env.ncol + j]), end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")     # 创建冰湖环境
    env = env.unwrapped     # 解封装才能访问状态转移矩阵P
    env.render()    # 环境渲染,通常是弹窗显示或打印出可视化的环境

    holes = set()   # 初始化为集合
    ends = set()

    """ 
    env.P
    {s: {a: [(p1, s_next1, r1, done), (p2, s_next2, r2, done), ...]}}
    """
    # print(type(env.P))  # dict
    # print(type(env.P[0]))   # dict
    for s in env.P:
        for a in env.P[s]:
            for trans in env.P[s][a]:
                if trans[2] == 1.0:     # 获得奖励为1，代表是目标
                    ends.add(trans[1])  # 添加索引，与状态对应
                if trans[3] == True:    # 代表游戏结束
                    holes.add(trans[1])
    holes = holes - ends    # 去除到达终点的情况
    print("冰洞的索引:", holes)
    print("目标的索引:", ends)

    # for a in env.P[14]:     # 查看目标左边一格的状态转移信息
    #     print(env.P[14][a])

    action_meaning = ['<', 'v', '>', '^']
    theta = 1e-5
    gamma = 0.9

    print()
    print("策略迭代：")
    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iteration()
    print_agent(agent, action_meaning, holes, ends)

    print()
    print("价值迭代：")
    agent = ValueIteration(env, theta, gamma)
    agent.value_iteration()
    print_agent(agent, action_meaning, holes, ends)
