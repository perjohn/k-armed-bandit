from bernoulli_bandit import BernoulliBandit


def test_pull():
    bandit = BernoulliBandit()
    result = bandit.pull()
    assert result == 0 or result == 1
