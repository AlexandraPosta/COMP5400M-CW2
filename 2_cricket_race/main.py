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

import os
import sys
from multiprocessing import Process

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from simulation import CricketSimulation
from agent import CricketAgent
from environment import CricketEnvironment
from speaker import Speaker


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
        _environment = CricketEnvironment(room_dim)
        _agent = CricketAgent(agent_location, speed=1.0)
        _agent_2 = CricketAgent()
        _environment.add_agent(_agent)
        _environment.add_agent(_agent_2)
        _source = Speaker(source_loc[0])
        _environment.add_source(_source)
        simulation = CricketSimulation(_environment, audio_path)
        simulation.play_simulation()

        # p = Process(target=simulation.play_simulation())
        # p.start()
        # p.join()

        i += 1


if __name__ == "__main__":
    main()
