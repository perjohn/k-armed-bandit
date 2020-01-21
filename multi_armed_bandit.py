from bernoulli_bandit import BernoulliBandit, NonStationaryBernoulliBandit


class MultiArmedBandit:

    def __init__(self, arms: int, stationary: bool = True):
        self.arms = arms
        self.bandits = []
        for i in range(0, arms):
            self.bandits.append(BernoulliBandit() if stationary else NonStationaryBernoulliBandit())
        self.best_arm = self._get_best_arm()

    def pull(self, k: int) -> (int, int, int):
        results = self.get_all_pull_results(k)
        optimal_action = 1 if self.best_arm == k else 0
        return results[k], self.calculate_regret(results, k), optimal_action

    def get_all_pull_results(self, k):
        results = []
        for i in range(0, self.arms):
            results.append(self.bandits[i].pull())
        return results

    @staticmethod
    def calculate_regret(pull_results, k) -> int:
        return max(pull_results) - pull_results[k]

    def _get_best_arm(self):
        result = 0
        max_probability = self.bandits[0].probability
        for i in range(1, self.arms):
            if self.bandits[i].probability > max_probability:
                result = i
                max_probability = self.bandits[i].probability
        return result
