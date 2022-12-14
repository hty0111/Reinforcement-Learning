"""
Description: 不同的求解策略
version: v1.0
Author: HTY
Date: 2022-11-24 21:20:35
"""

import numpy as np


class Solver:
    """ 多臂老虎机算法框架 """
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)   # 每根拉杆的尝试次数
        self.regret = 0     # 当前步的累计懊悔
        self.actions = []   # 每一步动作
        self.regrets = []   # 每一步累计懊悔

    def update_regret(self, k):
        """ 更新累计懊悔并保存 """
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        # print(self.bandit.best_idx, k)
        # print(self.bandit.probs[self.bandit.best_idx])
        # print(self.bandit.probs[k])
        self.regrets.append(self.regret)

    def run_one_step(self):
        """
        根据策略选择动作、根据动作获取奖励和更新期望奖励估值
        由每个策略具体实现
        """
        raise NotImplementedError

    def run(self, num_steps):
        """ 主循环 """
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):
    """ epsilon贪婪算法 """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)  # 初始化每根拉杆的期望奖励估值

    def run_one_step(self):
        if np.random.rand() < self.epsilon:
            k = np.random.randint(0, self.bandit.K) # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates)   # 选择期望奖励估值最大的拉杆
        reward = self.bandit.step(k)    # 执行一次动作后获得的奖励
        # print("reward: ", reward)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (reward - self.estimates[k])  # 更新选择拉杆的期望奖励估值
        # print("estimates: ", self.estimates)
        return k


class DecayingEpsilonGreedy(Solver):
    """ epsilon值随时间衰减的epsilon-贪婪算法 """
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:  # epsilon值随时间衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        reward = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (reward - self.estimates[k])
        return k


class UCB(Solver):
    """ 上届置信算法，upper confident bound """
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.totol_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.totol_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.totol_count) / (2 * (self.counts + 1)))    # 计算每根拉杆的上界置信
        k = np.argmax(ucb)
        reward = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (reward - self.estimates[k])
        return k


class ThompsonSampling(Solver):
    """ 汤普森采样算法 """
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self.ones = np.ones(self.bandit.K)  # 每根拉杆奖励为1的次数
        self.zeros = np.ones(self.bandit.K)  # 每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self.ones, self.zeros)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)

        self.ones[k] += r  # 更新Beta分布的第一个参数
        self.zeros[k] += (1 - r)  # 更新Beta分布的第二个参数
        return k

