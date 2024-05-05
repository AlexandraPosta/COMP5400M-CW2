import sys
import numpy as np
import math
from scipy.io import wavfile
from matplotlib import pyplot as plt, patches
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from multiprocessing import Process

# Assume DoaCNN is properly imported from the specified path
path_to_parent = 'c:/Users/Alex/source/repos/COMP5400M-CW2'
sys.path.insert(0, path_to_parent)
from doa import DoaCNN, DoaMUSIC

class CricketSimulation:
    def __init__(self, room_dim, audio_path, mic_locations, source_locations):
        self.room_dim = room_dim
        self.audio_path = audio_path
        self.mic_locations = mic_locations
        self.source_locations = source_locations
        self.simulation_done = False
        plt.rcParams.update({'figure.figsize': [6,6], 'figure.autolayout': True, 'font.size': 12})
        self.fig, self.ax = plt.subplots()
        self.setup_room()
        self.signal = None          # To be initialised in play_simulation
        self.CNN = None             # To be initialised in play_simulation
        self.trail_patches = []     # List to hold the patches for the trails

    def setup_room(self):
        self.ax.set_xlim([0, self.room_dim[0]+0.1])
        self.ax.set_ylim([0, self.room_dim[1]+0.1])
        # Setup sound source patches
        self.source_patches = [patches.Circle((source[0], source[1]), radius=0.8, facecolor='green', linewidth=5) for source in self.source_locations]
        for patch in self.source_patches:
            self.ax.add_patch(patch)

        # Include custom legend
        legend_elements = [Line2D([0], [0], marker='o', lw=2, color='green',   label='Sound Source'),\
                        Line2D([0], [0], marker='o', lw=2, color="black", label='Cricket')]
        self.ax.legend(handles=legend_elements, loc='upper left')
        self.mic_patches = self.ax.add_patch(patches.Circle((self.mic_locations), radius=0.3, facecolor='black'))

    def update_plot(self, frames):
        if self.simulation_done:
            return

        # Get predictions
        self.mic_locations = self.process(self.source_locations[0], self.mic_locations)

        # Draw the new microphone positions
        new_patch = patches.Circle((self.mic_locations), radius=0.3, facecolor='black')
        self.ax.add_patch(new_patch)
        self.trail_patches.append(new_patch)  # Keep track of it (optional)
        self.fig.canvas.draw()

    def process(self, source, mic):
        new_mic = [0, 0]
        if (source[0]-0.5 < mic[0] < source[0]+0.5 and 
            source[1]-0.3 < mic[1] < source[1]+0.5):
            self.simulation_done = True
        elif (mic[0] > self.room_dim[0]):
            self.simulation_done = True
        else:
            self.CNN = DoaCNN(self.room_dim, self.source_locations, mic, 0.1)
            self.CNN.get_room(self.signal)

            pred = self.CNN.get_prediction()
            pred_degree = max(set(pred), key=pred.count)
            pred_rad = math.pi - pred_degree * math.pi / 180

            x_align = mic[0] + 0.08 * np.cos(pred_rad)
            y_align = mic[1] + 0.08 * np.sin(pred_rad)

            new_mic = [x_align, y_align, 0]
        return new_mic
   
    def play_simulation(self):
        fs, self.signal = wavfile.read(self.audio_path)
        self.CNN = DoaCNN(self.room_dim, self.source_locations, self.mic_locations, 0.1)
        self.CNN.get_room(self.signal)
        self.anim = FuncAnimation(self.fig, self.update_plot, frames=None, repeat=False, blit=False)
        plt.show()

# Example usage
ROOM_DIM = [100., 100.]
AUDIO_PATH = 'c:/Users/Alex/source/repos/COMP5400M-CW2/sound_data/cricket.wav'
microphone = [ROOM_DIM[0]/2, 5, 0]
source_loc = [[ROOM_DIM[0]/2+25, ROOM_DIM[1]-10]]
simulation = CricketSimulation(ROOM_DIM, AUDIO_PATH, microphone, source_loc)

p = Process(target=simulation.play_simulation())
p.start()
