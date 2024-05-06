import os
import sys

from scipy.io import wavfile
from matplotlib import pyplot as plt, patches
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation


class CricketSimulation:
    def __init__(self, agent, environment, audio_path):
        self.agent = agent
        self.environment = environment
        self.audio_path = audio_path
        self.agent_location = self.agent.get_position()
        plt.rcParams.update(
            {"figure.figsize": [6, 6], "figure.autolayout": True, "font.size": 12}
        )
        self.fig, self.ax = plt.subplots()
        self.setup_room()
        self.setup_export_paths()
        self.signal = None  # To be initialised in play_simulation
        self.trail_patches = []  # List to hold the patches for the trails

    def update(self, frames):

        # Check if the agent has reached the sound source
        if self.agent.mate:
            self.anim.event_source.stop()
            plt.close(self.fig)
            return

        source = self.environment.get_source_locations()
        dims = self.environment.get_room_dimensions()
        # Get updates and draw the new positions
        if (
            source[0][0] - 0.3 < self.agent_location[0] < source[0][0] + 0.3
            and source[0][1] - 0.3 < self.agent_location[1] < source[0][1] + 0.3
        ):
            self.agent.mate = True
        elif self.agent_location[1] > dims[1]:
            self.agent.mate = True
        else:
            self.agent_location = self.agent.move(
                self.agent_location, dims, source, self.signal
            )
            new_patch = patches.Circle(
                (self.agent_location), radius=dims[0] / 300, facecolor="black"
            )
            self.ax.add_patch(new_patch)
            self.trail_patches.append(new_patch)
            self.fig.canvas.draw()

    def play_simulation(self):
        fs, self.signal = wavfile.read(self.audio_path)
        self.anim = FuncAnimation(
            self.fig, self.update, frames=None, repeat=False, blit=False
        )
        plt.show()

        # Save the png
        self.fig.savefig(self.png_path, dpi=300)
        # Save the gif
        self.anim.save(self.gif_path, writer="imagemagick", fps=1)

    def setup_room(self):
        dims = self.environment.get_room_dimensions()
        self.ax.set_xlim([0, dims[0] + 0.1])
        self.ax.set_ylim([0, dims[1] + 0.1])
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
        legend_elements = [
            Line2D([0], [0], marker="o", lw=0, color="green", label="Sound Source"),
            Line2D([0], [0], marker="o", lw=0, color="black", label="Cricket"),
        ]
        self.ax.legend(handles=legend_elements, loc="upper left")
        self.trail_patches = self.ax.add_patch(
            patches.Circle(
                (self.agent_location), radius=dims[0] / 300, facecolor="black"
            )
        )

    def setup_export_paths(self):
        # Get the path of the current .py file
        current_path = os.path.dirname(os.path.abspath(__file__))
        # Append the "output" folder to the current path
        current_path = os.path.join(current_path, "output")
        # Get the parent directory of the current path
        parent_path = os.path.join(os.path.dirname(current_path), "output")

        if not os.path.exists(current_path):
            os.makedirs(current_path)

        # See if there are any other files inside the "output" folder
        files = os.listdir(current_path)
        # If there are files, get the last file
        # The files are named as "output_gif_0.gif", "output_gif_1.gif", etc.
        last_number = 0
        if files:
            last_file = files[-1]
            # Get the number of the last file
            last_number = int(last_file.split("_")[-1].split(".")[0])
            # Increment the number
            last_number += 1

        # Add the parent path to the system path
        sys.path.insert(0, parent_path)

        self.gif_path = os.path.join(parent_path, f"output_gif_{last_number}.gif")
        self.png_path = os.path.join(parent_path, f"output_png_{last_number}.png")
