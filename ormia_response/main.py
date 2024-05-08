import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from cricket_lib import agent, environment, simulator, speaker

def main():
    # Room dimensions
    room_dim = [10.0, 10.0]
    audio_path = parent_dir + '/sound_data/cricket.wav'
    audio_path2 = parent_dir + '/sound_data/arctic_a0001.wav'
    output_path = parent_dir + '/ormia_response/data/'
    agent_location = [5, 1, 0]
    source_loc = [[room_dim[0]/2-2, room_dim[1]-1],[room_dim[0]/2+2, room_dim[1]-1]]

    # Agent and Simulation
    _agent = agent.CricketAgent(agent_location, speed=1.0, available_space=room_dim)
    _environment = environment.CricketEnvironment(room_dim)
    _source = speaker.Speaker(source_loc[0])
    _source2 = speaker.Speaker(source_loc[1])
    _environment.add_agent(_agent)
    _environment.add_source(_source)
    _environment.add_source(_source2)
    _simulator = simulator.CricketSimulation(_environment, [audio_path, audio_path2], output_path)
    _simulator.play_simulation()

if __name__ == "__main__":
    main()