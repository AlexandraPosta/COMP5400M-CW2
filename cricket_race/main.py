"""

COMP5400M - CW2
Author Name: Bogdan-Alexandru Ciurea - sc20bac

This is the main file for the compete agents.

The compete agents are agents that are trained to compete against each other in a game.

The rules of the game are as follows:
- The game is a 2-player game
- The game is a live simulation and will simulate the game in real-time
- The 2 players will rush for the closest sound source
- Sound sources will be placed randomly on the map but the catch is that the sound source can only be heard by ears which have a certain radius
- The sound can bang on the walls and the sound will be heard by the ears on the other side of the wall

	
"""

import numpy as np
from cricket_lib import agent, environment, simulator, speaker


def main():
    # Room dimensions
    room_dim = [10.0, 10.0]
    cricket_number = 4
    sound_source_number = 2
    i = 0

    # Setup the simulation

    audio_path = "./sound_data/cricket.wav"
    audio_paths = [audio_path for i in range(sound_source_number)]

    # Agent and Simulation
    _environment = environment.CricketEnvironment(room_dim)

    # Add agents to the environment
    for i in range(cricket_number):
        _environment.add_agent(
            agent.CricketAgentEvolution(
                [
                    np.random.uniform(0.5, room_dim[0] - 0.5),
                    np.random.uniform(0.5, room_dim[1] / 2 - 0.5),
                ]
            )
        )

    # Add sound source to the environment
    for i in range(sound_source_number):
        _environment.add_source(
            speaker.Speaker(
                [
                    np.random.uniform(0.5, room_dim[0] - 0.5),
                    np.random.uniform(room_dim[1] * 2 / 3 + 0.5, room_dim[1] - 0.5),
                ]
            )
        )

    simulation = simulator.CricketSimulation(
        _environment,
        audio_paths,
        "../cricket_race/",
    )

    while i < 5:
        try:
            simulation.play_simulation()

            # Find the winner
            if simulation.environment.has_winner():
                # Get the winner
                winner: agent.CricketAgentEvolution = simulation.environment.get_winner()
                winner.evolve()
                winner.move_to_random_position()
                simulation.environment.remove_looser_agents()
                simulation.environment.add_agent(winner)
                simulation.reset_environment()

            else :
                simulation.reset_environment()

        except Exception as e:
            print(e)
            simulation.reset_environment()

        # Simulation variables

        i += 1


if __name__ == "__main__":
    main()
