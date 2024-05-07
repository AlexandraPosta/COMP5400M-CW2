import os
import sys
from multiprocessing import Process

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from cricket_lib.agent import CricketAgent
from cricket_lib.speaker import Speaker
from cricket_lib.environment import CricketEnvironment
from cricket_lib.simulator import CricketSimulation

def main():
    # Room dimensions
    room_dim = [10., 10.]
    agent_location = [room_dim[0]/2, 1, 0]
    source_loc = [room_dim[0]/2+3, room_dim[1]-1]
    i = 1
    
    while (i < 5):
        # Simulation variables
        audio_path = './sound_data/cricket.wav'
        output_path = './3_cricket_behaviour/figures/cricket_vanilla_simulation_'
        output_path = output_path + str(i) + '.png'

        # Agent and Simulation
        _agent = CricketAgent(agent_location, speed=1.0)
        _environment = CricketEnvironment(room_dim)
        _source = Speaker(source_loc)
        _environment.add_agent(_agent)
        _environment.add_source(_source)
        simulation = CricketSimulation(_environment, audio_path, output_path)
        simulation.play_simulation()

        i+=1

if __name__ == "__main__":
    main()
