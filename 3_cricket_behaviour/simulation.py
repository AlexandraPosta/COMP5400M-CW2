from scipy.io import wavfile
from matplotlib import pyplot as plt, patches
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

class CricketSimulation:
    def __init__(self, agent, agent_location, environment, audio_path):
        self.agent = agent
        self.environment = environment
        self.audio_path = audio_path
        self.agent_location = agent_location
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
        self.source_patches = [patches.Circle((source[0], source[1]), radius=dims[0]/100, facecolor='green', linewidth=5) for source in self.environment.get_source_locations()]
        for patch in self.source_patches:
            self.ax.add_patch(patch)
        # Include custom legend
        legend_elements = [Line2D([0], [0], marker='o', lw=2, color='green', label='Sound Source'),\
                           Line2D([0], [0], marker='o', lw=2, color="black", label='Cricket')]
        self.ax.legend(handles=legend_elements, loc='upper left')
        self.trail_patches = self.ax.add_patch(patches.Circle((self.agent_location), radius=dims[0]/300, facecolor='black'))

    def update(self, frames):
        if self.agent.mate:
            return
        source = self.environment.get_source_locations()
        dims = self.environment.get_room_dimensions()
        # Get updates and draw the new positions
        if (source[0][0]-0.5 < self.agent_location[0] < source[0][0]+0.5 and
            source[0][1]-0.3 < self.agent_location[1] < source[0][1]+0.5):
            self.agent.mate = True
        elif (self.agent_location[0] > dims[0]):
            self.agent.mate = True
        else:
            self.agent_location = self.agent.move(self.agent_location, dims, source, self.signal, 0.1)
            new_patch = patches.Circle((self.agent_location), radius=dims[0]/300, facecolor='black')
            self.ax.add_patch(new_patch)
            self.trail_patches.append(new_patch)
            self.fig.canvas.draw()
  
    def play_simulation(self):
        fs, self.signal = wavfile.read(self.audio_path)
        self.anim = FuncAnimation(self.fig, self.update, frames=None, repeat=False, blit=False)
        plt.show()
