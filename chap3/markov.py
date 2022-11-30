"""
Description: 
version: v1.0
Author: HTY
Date: 2022-11-26 19:47:05
"""

import numpy as np


def compute_state_value(P, R, gamma):
    """ 状态价值函数：当前状态下所有动作的期望 """
    R = np.array(R).reshape((-1, 1))    # 写成列向量
    value = np.dot(np.linalg.inv(np.eye(np.array(P_mdp2mrp).shape[0]) - gamma * P), R)   # 贝尔曼方程的矩阵形式，V=(I-gamma*P)^(-1)*R
    return value


def compute_action_value(P, R, V, gamma):
    """ 动作价值函数：当前状态下每个动作的价值 """
    Q = np.zeros((np.array(P).shape[0], np.array(A).shape[0]))  # 横坐标为状态，纵坐标为动作
    # TODO, first transform A & R into matrix
    return Q


def sample(MDP, Pi, timestep_max, sample_number):
    """
    采样不同的序列
    :param MDP: 元组
    :param timestep_max: 最长采样的实践步
    :param sample_number: 需要采样的序列数量
    :return:
    """
    S, A, P, R, gamma = MDP
    episodes = []   # 采样得到的不同序列
    for _ in range(sample_number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]     # 随机选一个除s5以外的状态作为起点
        while s != "s5" and timestep < timestep_max:
            timestep += 1
            a, r, s_next = 0, 0, 0

            # 随机采样动作
            rand, temp = np.random.rand(), 0
            for a_opt in A:
                temp += Pi.get("-".join([s, a_opt]), 0)     # 当前状态下，每个动作的概率求和为1
                if temp > rand:     # 随机数rand落在哪个动作的概率区间就选择这个动作
                    a = a_opt
                    r = R["-".join([s, a])]
                    break

            # 随机采样状态
            rand, temp = np.random.rand(), 0
            for s_opt in S:
                temp += P.get("-".join(["-".join([s, a]), s_opt]), 0)
                if temp > rand:
                    s_next = s_opt
                    break

            episode.append((s, a, r, s_next))   # 把（s,a,r,s_next）元组放入序列中
            s = s_next  # s_next变成当前状态，开始下一次循环
        episodes.append(episode)
    return episodes


def monte_carlo(episodes, V, N, gamma):
    """
    蒙特卡洛近似求状态价值
    :param episodes: 状态动作序列
    :param V: 每个状态的价值
    :param N: 每个状态的采样次数
    :param gamma: 折扣因子
    :return:
    """
    for episode in episodes:
        G = 0   # 回报return
        for i in range(len(episode) - 1, -1, -1):   # 从后往前计算
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1     # 更新计数器
            V[s] = V[s] + (G - V[s]) / N[s]     # 增量式更新状态价值


def occupancy(episodes, s, a, timestep_max, gamma):
    """ 通多采样很多轨迹来计算状态动作对（s, a）出现的频率，以此估算占用度量 """
    rho = 0
    total_times = np.zeros(timestep_max)    # 记录每个时间步t被经历的次数
    occur_times = np.zeros(timestep_max)    # 记录状态动作对为（s, a）的次数
    for episode in episodes:
        for i in range(len(episode)):
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    for i in reversed(range(timestep_max)):
        if total_times[i]:
            rho += gamma**i * occur_times[i] / total_times[i]
    return (1 - gamma) * rho


if __name__ == "__main__":
    S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
    A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]    # 动作集合
    # 状态转移函数，环境带来的不确定性，状态更新是基于状态转移函数的随机采样
    P = {
        "s1-保持s1-s1": 1.0,
        "s1-前往s2-s2": 1.0,
        "s2-前往s1-s1": 1.0,
        "s2-前往s3-s3": 1.0,
        "s3-前往s4-s4": 1.0,
        "s3-前往s5-s5": 1.0,
        "s4-前往s5-s5": 1.0,
        "s4-概率前往-s2": 0.2,
        "s4-概率前往-s3": 0.4,
        "s4-概率前往-s4": 0.4,
        "s5-前往s5-s5": 1.0
    }
    # 奖励函数
    R = {
        "s1-保持s1": -1,
        "s1-前往s2": 0,
        "s2-前往s1": -1,
        "s2-前往s3": -2,
        "s3-前往s4": -2,
        "s3-前往s5": 0,
        "s4-前往s5": 10,
        "s4-概率前往": 1,
        "s5-前往s5": 0
    }
    gamma = 0.5     # 折扣因子
    MDP = (S, A, P, R, gamma)   # 马尔可夫决策过程

    # 策略1，随机策略，策略是动作的概率分布，动作是依据这一分布的随机采样
    Pi_1 = {
        "s1-保持s1": 0.5,
        "s1-前往s2": 0.5,
        "s2-前往s1": 0.5,
        "s2-前往s3": 0.5,
        "s3-前往s4": 0.5,
        "s3-前往s5": 0.5,
        "s4-前往s5": 0.5,
        "s4-概率前往": 0.5,
        "s5-前往s5": 1.0
    }

    # 策略2
    Pi_2 = {
        "s1-保持s1": 0.6,
        "s1-前往s2": 0.4,
        "s2-前往s1": 0.3,
        "s2-前往s3": 0.7,
        "s3-前往s4": 0.5,
        "s3-前往s5": 0.5,
        "s4-前往s5": 0.1,
        "s4-概率前往": 0.9,
        "s5-前往s5": 1.0
    }

    """ 求策略1下的马尔可夫奖励过程 """
    Pi = Pi_1

    P_mdp2mrp = np.zeros((len(S), len(S)))    # 马尔可夫奖励过程的状态转移函数，即利用策略pi对所有动作a求期望
    for p in P:
        s, a, s_next = p.split("-")     # 当前状态，动作，执行动作后的状态
        pi = "-".join([s, a])   # 当前状态的执行的动作，即某种策略
        P_mdp2mrp[S.index(s)][S.index(s_next)] = P.get(p) * Pi.get(pi)
    # print(P_mdp2mrp)

    R_mdp2mrp = np.zeros(len(S))
    for r in R:
        s, a = r.split("-")
        R_mdp2mrp[S.index(s)] += R.get(r) * Pi.get(r)
    # print(R_mdp2mrp)

    """ 状态价值的解析解 """
    V = compute_state_value(P_mdp2mrp, R_mdp2mrp, gamma)
    # print(V)

    # Q = compute_action_value(P_mdp2mrp, R_mdp2mrp, V, gamma)
    # print(Q)

    """ 状态价值的蒙特卡洛近似解 """
    episodes = sample(MDP, Pi_1, 20, 1000)
    # print(episodes[0])

    V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
    N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
    monte_carlo(episodes, V, N, gamma)
    # print(V)

    """ 动作状态对的占用度量，即不同策略下的概率密度函数 """
    episodes_1 = sample(MDP, Pi_1, 1000, 1000)
    episodes_2 = sample(MDP, Pi_2, 1000, 1000)
    rho_1 = occupancy(episodes_1, "s4", "概率前往", 1000, gamma)
    rho_2 = occupancy(episodes_2, "s4", "概率前往", 1000, gamma)
    # print(rho_1, rho_2)
