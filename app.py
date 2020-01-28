import click
import matplotlib.pyplot as plt
import numpy as np

from agent import Agent


@click.command()
@click.option('--arms', default=5, help='Number of arms.', type=int)
@click.option('--steps', default=100, help='Number of steps.', type=int)
@click.option('--runs', default=1, help='Number of runs.', type=int)
@click.option('--exploration-rate', default=0.1, type=float)
@click.option('--initial-values', default=0., type=float)
@click.option('--stationary/--non-stationary', default=True)
def grid(arms, steps, runs, exploration_rate, initial_values, stationary):
    agent = Agent(arms=arms, runs=runs, steps=steps, exploration_rate=exploration_rate, initial_values=initial_values,
                  stationary=stationary)
    agent.play()
    optimal_averages = np.average(agent.action_optimal, axis=0)
    plt.plot(100 * optimal_averages)
    plt.ylabel('% Optimal action')
    plt.xlabel('Steps')
    plt.title(f'Exploration rate: {exploration_rate}, initial values: {initial_values}')
    plt.show()


if __name__ == '__main__':
    grid()
