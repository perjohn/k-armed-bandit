from mock import patch
import pytest

from agent import Agent


@patch('numpy.random.random')
def test_agent(mock_random):
    mock_random.side_effect = [0.2, 0.3]
    agent = Agent(arms=2)
    assert agent.maximum_average_reward == 0.3


def test_update_averages():
    agent = Agent(arms=2)
    agent.update_averages(1, 0, 1, 0)
    assert agent.average_reward == 1
    assert agent.average_regret == 0
    assert agent.optimal_action_percentage == 0

    agent.update_averages(0, 1, 2, 0)
    assert agent.average_reward == 0.5
    assert agent.average_regret == 0.5
    assert agent.optimal_action_percentage == 0

    agent.update_averages(1, 0, 3, 1)
    assert agent.average_reward == pytest.approx(0.667, abs=0.001)
    assert agent.average_regret == pytest.approx(0.333, abs=0.001)
    assert agent.optimal_action_percentage == pytest.approx(0.333, abs=0.001)
