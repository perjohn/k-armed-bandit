import numpy as np


class BernoulliBandit:
    def __init__(self):
        self.probability = np.random.random()

    def pull(self):
        return np.random.binomial(1, self.probability)


class NonStationaryBernoulliBandit:
    def __init__(self):
        self.probability = np.random.random()

    def pull(self):
        result = np.random.binomial(1, self.probability)
        self.set_probability()
        return result

    def set_probability(self):
        self.probability = self.probability + np.random.normal(0, 0.01)
        self.probability = max(0, self.probability)
        self.probability = min(self.probability, 1)
