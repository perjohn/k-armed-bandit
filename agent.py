import numpy as np
from tqdm import tqdm

from multi_armed_bandit import MultiArmedBandit


class Agent:
    def __init__(self, arms: int, runs: int, steps: int, exploration_rate: float, initial_values: float,
                 stationary: bool = True):
        self.arms = arms
        self.runs = runs
        self.steps = steps
        self.initial_values = initial_values
        self.stationary = stationary
        self.multi_armed_bandit = None
        self.average_regret = 0
        self.average_reward = 0
        self.maximum_average_reward = 0
        self.exploration_rate = exploration_rate
        self.value_estimates = self.init_value_estimates()
        self.action_optimal = np.zeros((runs, steps))
        self.rewards = np.zeros((runs, steps))

    def play(self):
        for run_counter in tqdm(range(0, self.runs)):
            self.multi_armed_bandit = MultiArmedBandit(arms=self.arms, stationary=self.stationary)
            self.value_estimates = self.init_value_estimates()
            self._run(run_counter=run_counter)

    def _run(self, run_counter: int):
        # self.print_bandit_info()
        for step_counter in range(0, self.steps):
            arm = self._choose_arm()
            reward, regret, optimal = self.multi_armed_bandit.pull(arm)
            self.update_value_estimate(reward=reward, arm=arm, pull=step_counter + 1)
            self.action_optimal[run_counter][step_counter] = optimal
            self.rewards[run_counter][step_counter] = reward
            # self.exploration_rate = 1 / math.log(i + 0.00001)
        # print([f'{value:1.3f}' for value in self.value_estimates])
        # print('-----------------------------------------------------------')

    def _choose_arm(self) -> int:
        if np.random.uniform(0, 1) <= self.exploration_rate:
            return np.random.randint(0, self.arms)
        else:
            return int(np.argmax(self.value_estimates))
            # return np.random.choice(np.arange(0, self.arms), p=self.value_estimates / np.sum(self.value_estimates))

    def print_bandit_info(self):
        print([f'{bandit.probability:1.3f}' for bandit in self.multi_armed_bandit.bandits])

    def update_value_estimate(self, reward: int, arm: int, pull: int):
        self.value_estimates[arm] = self.value_estimates[arm] + (reward - self.value_estimates[arm]) / pull

    def init_value_estimates(self) -> np.ndarray:
        return np.full((self.arms,), self.initial_values)
