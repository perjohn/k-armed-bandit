from bernoulli_bandit import BernoulliBandit, NonStationaryBernoulliBandit


class MultiArmedBandit:

    def __init__(self, arms: int, stationary: bool = True):
        self.arms = arms
        self.bandits = []
        for i in range(0, arms):
            self.bandits.append(BernoulliBandit() if stationary else NonStationaryBernoulliBandit())

    def pull(self, k: int) -> (int, int):
        results = self.get_all_pull_results(k)
        return results[k], self.calculate_regret(results, k)

    def get_all_pull_results(self, k):
        results = []
        for i in range(0, self.arms):
            results.append(self.bandits[k].pull())
        return results

    @staticmethod
    def calculate_regret(pull_results, k) -> int:
        return max(pull_results) - pull_results[k]
