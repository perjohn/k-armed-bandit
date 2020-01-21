from mock import patch

from multi_armed_bandit import MultiArmedBandit


@patch('numpy.random.random')
def test_init(mock_random):
    mock_random.side_effect = [0.2, 0.3]
    multi_armed_bandit = MultiArmedBandit(2)
    assert multi_armed_bandit.best_arm == 1

    mock_random.side_effect = [0.2, 0.1]
    multi_armed_bandit = MultiArmedBandit(2)
    assert multi_armed_bandit.best_arm == 0


@patch('numpy.random.random')
@patch('numpy.random.binomial')
def test_pull_optimal(mock_binomial, mock_random):
    mock_random.side_effect = [0.2, 0.3]
    mock_binomial.side_effect = [0, 1]
    multi_armed_bandit = MultiArmedBandit(2)
    result, regret, optimal = multi_armed_bandit.pull(1)
    assert result == 1
    assert regret == 0
    assert optimal == 1


@patch('numpy.random.random')
@patch('numpy.random.binomial')
def test_pull_not_optimal(mock_binomial, mock_random):
    mock_random.side_effect = [0.2, 0.3]
    mock_binomial.side_effect = [1, 0]
    multi_armed_bandit = MultiArmedBandit(2)
    result, regret, optimal = multi_armed_bandit.pull(0)
    assert result == 1
    assert regret == 0
    assert optimal == 0


def test_calculate_regret():
    multi_armed_bandit = MultiArmedBandit(2)
    result = multi_armed_bandit.calculate_regret([0, 1], 1)
    assert result == 0

    result = multi_armed_bandit.calculate_regret([0, 1], 0)
    assert result == 1
