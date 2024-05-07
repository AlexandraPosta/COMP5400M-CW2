import os
import sys

from scipy.io import wavfile
from matplotlib import pyplot as plt, patches
from matplotlib.animation import FuncAnimation
import concurrent.futures

from environment import CricketEnvironment


class CricketSimulation:
    def __init__(self, environment: CricketEnvironment, audio_path):
        self.environment: CricketEnvironment = environment
        self.audio_path = audio_path
        plt.rcParams.update(
            {"figure.figsize": [6, 6], "figure.autolayout": True, "font.size": 12}
        )
        self.fig, self.ax = plt.subplots()
        self.setup_room()
        self.setup_export_paths()
        self.signal = None  # To be initialised in play_simulation
        self.trail_patches = []  # List to hold the patches for the trails

    def update(self, _):

        # Check if any agent has reached the sound source
        if any(
            agent.check_mate(self.environment.get_source_locations())
            for agent in self.environment.agents
        ):
            self.anim.event_source.stop()
            plt.close(self.fig)
            return

        sources = self.environment.get_source_locations()
        dimensions = self.environment.get_room_dimensions()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for agent in self.environment.agents:
                future = executor.submit(agent.move, dimensions, sources, self.signal)
                futures.append(future)

            # Wait for all futures to complete
            concurrent.futures.wait(futures)

        # Draw the new position of the agent
        for agent in self.environment.get_agent_locations():
            new_patch = patches.Circle(
                (agent), radius=dimensions[0] / 300, facecolor="black"
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

        # Setup agent patches
        self.trail_patches = [
            patches.Circle(
                position, radius=self.environment.room_dim[0] / 300, facecolor="black"
            )
            for position in self.environment.get_agent_locations()
        ]
        for patch in self.trail_patches:
            self.ax.add_patch(patch)

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
