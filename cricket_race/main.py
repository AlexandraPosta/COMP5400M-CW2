"""
This is the main file for the compete agents.

The compete agents are agents that are trained to compete against each other in a game.

The rules of the game are as follows:
- The game is a 2-player game
- The game is a live simulation and will simulate the game in real-time
- The 2 players will rush for the closest sound source
- Sound sources will be placed randomly on the map but the catch is that the sound source can only be heard by ears which have a certain radius
- The sound can bang on the walls and the sound will be heard by the ears on the other side of the wall

"""

from cricket_lib import agent, environment, simulator, speaker


def main():
    # Room dimensions
    room_dim = [10.0, 10.0]
    agent_location = [room_dim[0] / 2, 1, 0]
    source_loc = [[3, 3]]
    i = 0

    while i < 1:
        # Simulation variables
        audio_path = "./sound_data/cricket.wav"

        # Agent and Simulation
        _environment = environment.CricketEnvironment(room_dim)
        _agent_1 = agent.CricketAgent()
        _agent_2 = agent.CricketAgent()
        _environment.add_agent(_agent_1)
        _environment.add_agent(_agent_2)
        _source = speaker.Speaker()
        _environment.add_source(_source)
        simulation = simulator.CricketSimulation(
            _environment, [audio_path], "../cricket_race/"
        )
        simulation.play_simulation()
        i += 1


if __name__ == "__main__":
    main()
