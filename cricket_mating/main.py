import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from cricket_lib import agent, environment, simulator, speaker


def main():
    # Room dimensions
    room_dim = [10.0, 10.0]
    agent_location = [room_dim[0] / 2, 1, 0]
    source_loc = [[room_dim[0] / 2 - 3, room_dim[1] - 4],
                  [room_dim[0] / 2 + 0.5, room_dim[1] - 1.2]]

    # Simulation variables
    audio_path = "./sound_data/cricket.wav"
    output_path = "../cricket_mating/memory/"

    # Agent and Simulation
    _agent = agent.CricketAgent(position=agent_location, available_space=room_dim)
    _environment = environment.CricketEnvironment(room_dim)
    _source_1 = speaker.Speaker(source_loc[0])
    _environment.add_agent(_agent)
    _environment.add_source(_source_1)
    _simulator = simulator.CricketSimulation(_environment, [audio_path, audio_path], output_path)
    _simulator.play_simulation()

if __name__ == "__main__":
    main()
