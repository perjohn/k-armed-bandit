from mock import patch
import numpy as np

from agent import Agent


@patch('numpy.random.random')
def test_agent(mock_random):
    mock_random.side_effect = [0.2, 0.3]
    agent = Agent(arms=2, runs=1, steps=5, exploration_rate=0.1, initial_values=0.)
    assert np.array_equal(agent.value_estimates, np.array([0., 0.]))


@patch('numpy.random.random')
@patch('numpy.random.uniform')
@patch('numpy.random.randint')
@patch('numpy.random.binomial')
def test_play_one_run(mock_binomial, mock_randint, mock_uniform, mock_random):
    mock_random.side_effect = [0.2, 0.3]
    mock_uniform.side_effect = [0.2, 0.05]
    mock_randint.side_effect = [1]
    mock_binomial.side_effect = [0, 1, 0, 1]
    agent = Agent(arms=2, runs=1, steps=2, exploration_rate=0.1, initial_values=0.)
    agent.play()
    assert np.array_equal(agent.action_optimal, np.array([[0, 1]]))
    assert np.array_equal(agent.value_estimates, np.array([0., 1.]))


@patch('numpy.random.random')
@patch('numpy.random.uniform')
@patch('numpy.random.randint')
@patch('numpy.random.binomial')
def test_play_two_runs(mock_binomial, mock_randint, mock_uniform, mock_random):
    mock_random.side_effect = [0.2, 0.3, 0.8, 0.1]
    mock_uniform.side_effect = [0.2, 0.05, 0.2, 0.2]
    mock_randint.side_effect = [1]
    mock_binomial.side_effect = [0, 1, 0, 1, 0, 0, 1, 0]
    agent = Agent(arms=2, runs=2, steps=2, exploration_rate=0.1, initial_values=0.)
    agent.play()
    assert np.array_equal(agent.action_optimal, np.array([[0, 1], [1, 1]]))
