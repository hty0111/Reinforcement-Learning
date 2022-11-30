"""
Description: 多臂老虎机
version: v1.0
Author: HTY
Date: 2022-11-24 21:07:04
"""

import numpy as np
from matplotlib import pyplot as plt
from solver import EpsilonGreedy, DecayingEpsilonGreedy, UCB, ThompsonSampling



class Bandit:
    """多臂老虎机"""
    def __init__(self, K):
        self.probs = np.random.uniform(low=0, high=1, size=K)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


def plot_results(solvers, solver_names):
    """
    生成累积懊悔随时间变化的图像
    :param solvers: 是一个列表，其中的每个元素是一种特定的策略
    :param solver_names: 是一个列表，存储每个策略的名称
    :return:
    """
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    np.random.seed(1)
    bandit = Bandit(10)

    # epsilon_greedy_solver = EpsilonGreedy(bandit)
    # epsilon_greedy_solver.run(5000)
    # plot_results([epsilon_greedy_solver], "Epsilon greedy")

    epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    epsilon_greedy_solver_list = [EpsilonGreedy(bandit, epsilon=e) for e in epsilons]
    epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
    for solver in epsilon_greedy_solver_list:
        solver.run(5000)
    # plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit)
    decaying_epsilon_greedy_solver.run(5000)
    # plot_results([decaying_epsilon_greedy_solver], "Decaying epsilon")

    ucb_solver = UCB(bandit, 1)
    ucb_solver.run(5000)
    # plot_results([ucb_solver], "UCB")

    thompson_solver = ThompsonSampling(bandit)
    thompson_solver.run(5000)
    plot_results([thompson_solver], "Thompson")






