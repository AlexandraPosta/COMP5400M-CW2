import os
import sys
from multiprocessing import Process

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from simulation import CricketSimulation
from agent import CricketAgent
from environment import CricketEnvironment

def main():
    # Room dimensions
    room_dim = [10., 10.]
    agent_location = [room_dim[0]/2, 1, 0]
    source_loc = [[room_dim[0]/2+3, room_dim[1]-1]]
    i = 1
    
    while (i < 16):
        # Simulation variables
        audio_path = './sound_data/cricket.wav'
        output_path = './3_cricket_behaviour/figures/cricket_vanilla_simulation_'
        output_path = output_path + str(i) + '.png'

        # Agent and Simulation
        _agent = CricketAgent()
        _environment = CricketEnvironment(room_dim, source_loc)
        simulation = CricketSimulation(_agent, agent_location, _environment, audio_path, output_path)

        p = Process(target=simulation.play_simulation())
        p.start()
        p.join()

        i+=1

if __name__ == "__main__":
    main()
