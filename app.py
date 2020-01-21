import click

from agent import Agent


@click.command()
@click.option('--arms', default=5, help='Number of arms.', type=int)
@click.option('--rounds', default=100, help='Number of learning rounds.', type=int)
@click.option('--stationary/--non-stationary', default=True)
def grid(arms: int, rounds: int, stationary: bool):
    agent = Agent(arms=arms, stationary=stationary)
    agent.run(rounds=rounds)


if __name__ == '__main__':
    grid()
