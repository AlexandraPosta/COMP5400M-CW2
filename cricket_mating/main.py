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
    source_loc = [room_dim[0] / 2 + 3, room_dim[1] - 1]
    i = 0

    # Simulation variables
    audio_path = "./sound_data/cricket.wav"
    output_path = "../cricket_mating/memory/"

    while i < 1:
        # Agent and Simulation
        _agent = agent.CricketAgent(agent_location, speed=1.0, available_space=room_dim)
        _environment = environment.CricketEnvironment(room_dim)
        _source = speaker.Speaker(source_loc)
        _environment.add_agent(_agent)
        _environment.add_source(_source)
        _simulator = simulator.CricketSimulation(_environment, [audio_path], output_path)
        _simulator.play_simulation()
        i += 1

if __name__ == "__main__":
    main()
