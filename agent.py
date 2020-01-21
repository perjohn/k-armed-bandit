import math

import numpy as np

from multi_armed_bandit import MultiArmedBandit


class Agent:
    def __init__(self, arms: int, stationary: bool = True):
        self.arms = arms
        self.multi_armed_bandit = MultiArmedBandit(arms=arms, stationary=stationary)
        self.average_regret = 0
        self.average_reward = 0
        self.maximum_average_reward = 0
        self.set_maximum_reward()
        self.exploration_rate = 0.2
        self.value_estimates = np.full((arms,), 1 / arms)

    def run(self, rounds: int):
        self.print_bandit_info()
        for i in range(0, rounds):
            arm = self.choose_arm()
            reward, regret = self.multi_armed_bandit.pull(arm)
            self.update_averages(reward=reward, regret=regret, pull=i + 1)
            self.update_value_estimate(reward=reward, arm=arm, pull=i + 1)
            # self.exploration_rate = 1 / math.log(i + 0.00001)
        print(
            f'Result = {reward}, reward = {reward}, average reward = {self.average_reward:1.3f}, average regret = {self.average_regret:1.3f}')
        print(self.value_estimates)

    def choose_arm(self) -> int:
        if np.random.uniform(0, 1) <= self.exploration_rate:
            return np.random.randint(0, self.arms)
        else:
            return int(np.argmax(self.value_estimates))
            # return np.random.choice(np.arange(0, self.arms), p=self.value_estimates / np.sum(self.value_estimates))

    def print_bandit_info(self):
        for i in range(0, self.arms):
            bandit = self.multi_armed_bandit.bandits[i]
            print(f'Bandit {i:2.0f}: probability = {bandit.probability:1.3f}')
        print(f'Maximum average reward = {self.maximum_average_reward:1.3f}')

    def update_value_estimate(self, reward: int, arm: int, pull: int):
        self.value_estimates[arm] = self.value_estimates[arm] + (reward - self.value_estimates[arm]) / pull

    def update_averages(self, reward: int, regret: int, pull: int):
        self.average_reward = self.average_reward + (reward - self.average_reward) / pull
        self.average_regret = self.average_regret + (regret - self.average_regret) / pull

    def set_maximum_reward(self):
        self.maximum_average_reward = max([bandit.probability for bandit in self.multi_armed_bandit.bandits])
