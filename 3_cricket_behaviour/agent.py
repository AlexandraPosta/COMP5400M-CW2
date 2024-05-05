import math
import numpy as np
import sys

# Assume DoaCNN is properly imported from the specified path
path_to_parent = 'c:/Users/Alex/source/repos/COMP5400M-CW2'
sys.path.insert(0, path_to_parent)
from doa import DoaCNN

class CricketAgent:
    def __init__(self):
        self.mate = False
        self.auditory_sense = None

    def sense(self, position, room_dim, sound_sources, signal, snr):
        self.auditory_sense = DoaCNN(room_dim, sound_sources, position, snr)
        self.auditory_sense.get_room(signal)
        pred = self.auditory_sense.get_prediction()
        pred_degree = max(set(pred), key=pred.count)
        return math.pi - pred_degree * math.pi / 180

    def move(self, location, room_dim, sound_sources, signal, snr):
        direction = self.sense(location, room_dim, sound_sources, signal, snr)
        x_align = location[0] + 0.08 * np.cos(direction)
        y_align = location[1] + 0.08 * np.sin(direction)
        return [x_align, y_align, 0]
