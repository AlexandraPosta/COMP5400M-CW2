import os
import sys

from scipy.io import wavfile
from matplotlib import pyplot as plt, patches
from matplotlib.animation import FuncAnimation
import concurrent.futures

from .environment import CricketEnvironment


class CricketSimulation:
    def __init__(
        self, environment: CricketEnvironment, audio_path: str, destination_path: str
    ):
        """
        Initialize the simulation

        Args:
            environment (CricketEnvironment): The environment object
            audio_path (str): The path to the audio file
        """

        self.environment: CricketEnvironment = environment
        self.audio_path = audio_path
        plt.rcParams.update(
            {"figure.figsize": [6, 6], "figure.autolayout": True, "font.size": 12}
        )
        self.fig, self.ax = plt.subplots()
        self.setup_room()
        self.setup_export_paths(destination_path)
        self.signal = None  # To be initialised in play_simulation
        self.trail_patches = []  # List to hold the patches for the trails

    def update(self, _):
        """
        Update the positions of the agents and sound sources
        """

        # Check if any agent has reached the sound source or if any agent is under any sound source
        if (
            any(
                agent.check_mate(self.environment.get_source_locations())
                for agent in self.environment.agents
            )
            or self.environment.check_one_agent_under_source()
        ):
            # Close the figure
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

    def __animation_finished(self):
        """
        Callback function to handle animation finish event
        """
        self.anim.event_source.stop()  # Stop animation event source

    def play_simulation(self):
        """
        Play the simulation
        """

        fs, self.signal = wavfile.read(self.audio_path)
        self.anim = FuncAnimation(
            self.fig, self.update, frames=None, repeat=False, blit=False
        )
        self.fig.canvas.mpl_connect("close_event", self.__animation_finished)
        plt.show()

        # Save the png
        print(f"-------------- Saving the png at {self.png_path} --------------")
        self.fig.savefig(self.png_path, dpi=300)
        # Save the gif
        print(f"-------------- Saving the gif at {self.gif_path} --------------")
        self.anim.save(self.gif_path, writer="imagemagick", fps=2)

    def setup_room(self):
        """
        Setup the room with sound sources and agents
        """

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

    def setup_export_paths(self, destination_path: str) -> None:
        """
        Setup the paths for exporting the gif and png files

        Args:
            destination_path (str): The folder to save the files to

        The files are named as "output_gif_0.gif", "output_gif_1.gif", etc.
        or "output_png_0.png", "output_png_1.png", etc. and will be saved in the "output" folder in the parent directory of the current path
        """

        # Get the path of where the execution is happening
        current_path = os.path.dirname(os.path.realpath(__file__))
        # Append the "destination_path" to the current path
        current_path = os.path.join(current_path, destination_path)
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
