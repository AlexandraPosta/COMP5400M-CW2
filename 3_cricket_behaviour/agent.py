import math
import numpy as np
import sys

from doa import DoaCNN

class CricketAgent:
    def __init__(self):
        self.mate = False
        self.auditory_sense = None
        self.distance_mic = 0.1
        self.snr = 0.1

    def sense(self, position, room_dim, sound_sources, signal):
        self.auditory_sense = DoaCNN(room_dim, sound_sources, position, self.distance_mic, self.snr)
        self.auditory_sense.get_room(signal)
        pred = self.auditory_sense.get_prediction()
        pred_degree = max(set(pred), key=pred.count)
        return math.pi - pred_degree * math.pi / 180

    def move(self, location, room_dim, sound_sources, signal):
        direction = self.sense(location, room_dim, sound_sources, signal)
        x_align = location[0] + 0.08 * np.cos(direction)
        y_align = location[1] + 0.08 * np.sin(direction)
        return [x_align, y_align, 0]
