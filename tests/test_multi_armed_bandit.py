from multi_armed_bandit import MultiArmedBandit


def test_pull():
    multi_armed_bandit = MultiArmedBandit(2)
    result, regret = multi_armed_bandit.pull(0)
    assert result == 0 or result == 1
    assert regret == 0 or regret == 1

    result, regret = multi_armed_bandit.pull(1)
    assert result == 0 or result == 1
    assert regret == 0 or regret == 1


def test_calculate_regret():
    multi_armed_bandit = MultiArmedBandit(2)
    result = multi_armed_bandit.calculate_regret([0, 1], 1)
    assert result == 0

    result = multi_armed_bandit.calculate_regret([0, 1], 0)
    assert result == 1
