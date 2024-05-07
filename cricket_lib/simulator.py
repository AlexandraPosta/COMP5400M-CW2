import os
import sys

from scipy.io import wavfile
from matplotlib import pyplot as plt, patches
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import concurrent.futures

from cricket_lib.environment import CricketEnvironment

class CricketSimulation:
    def __init__(self, environment: CricketEnvironment, audio_path, output_path):
        self.environment: CricketEnvironment = environment
        self.audio_path = audio_path
        self.output_path = output_path
        plt.rcParams.update({'figure.figsize': [6,6], 'figure.autolayout': True, 'font.size': 12})
        self.fig, self.ax = plt.subplots()
        self.setup_room()
        self.signal = None          # To be initialised in play_simulation
        self.trail_patches = []     # List to hold the patches for the trails

    def setup_room(self):
        dims = self.environment.get_room_dimensions()
        self.ax.set_xlim([0, dims[0]+0.1])
        self.ax.set_ylim([0, dims[1]+0.1])

        # Setup sound source patches
        self.source_patches = [
            patches.Circle(
                (source[0], source[1]),
                radius=dims[0] / 100,
                facecolor="green",
                linewidth=5,
            )
            for source in self.environment.get_source_locations()
        ]
        for patch in self.source_patches:
            self.ax.add_patch(patch)

        # Include custom legend
        legend_elements = [Line2D([0], [0], marker='o', lw=0, color='green', label='Sound Source'),\
                           Line2D([0], [0], marker='o', lw=0, color='black', label='Cricket')]
        self.ax.legend(handles=legend_elements, loc='upper left')

        # Setup agent patches
        self.trail_patches = [
            patches.Circle(
                position, radius=self.environment.room_dim[0] / 300, facecolor="black"
            )
            for position in self.environment.get_agent_locations()
        ]
        for patch in self.trail_patches:
            self.ax.add_patch(patch)

    def update(self, _):
        # Check if any agent has reached the sound source
        if any(
            agent.check_mate(self.environment.get_source_locations())
            for agent in self.environment.agents
        ):
            self.fig.savefig(self.output_path)
            self.anim.event_source.stop()
            plt.close(self.fig)
            return

        source = self.environment.get_source_locations()
        dimensions = self.environment.get_room_dimensions()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for agent in self.environment.agents:
                future = executor.submit(agent.move, dimensions, source, self.signal)
                futures.append(future)

            # Wait for all futures to complete
            concurrent.futures.wait(futures)

        # Draw the new position of the agent
        for i in range(self.environment.agents.__len__()):
            for source in self.environment.get_source_locations():
                # Get updates and draw the new positions
                position = self.environment.agents[i].get_position()
                if (source[0]-0.2 < position[0] < source[0]+0.2 and
                    source[1]-0.2 < position[1] < source[1]+0.2):
                    self.environment.agents[i].mate = True
                elif (position[1] > dimensions[1]):
                    self.environment.agents[i].mate = True

            if any (self.environment.agents[i].mate for i in range(self.environment.agents.__len__())):
                self.environment.agents[i].move(
                    dimensions, self.environment.get_source_locations(), self.signal
                    )
                new_patch = patches.Circle(
                    position, radius=dimensions[0] / 300, facecolor="black"
                )
                self.ax.add_patch(new_patch)
                self.trail_patches.append(new_patch)
                self.fig.canvas.draw()
  
    def play_simulation(self):
        fs, self.signal = wavfile.read(self.audio_path)
        self.anim = FuncAnimation(self.fig, self.update, frames=None, repeat=False, blit=False)
        plt.show()
