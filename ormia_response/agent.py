import math
import numpy as np
import sys

from doa import DoaCNN, DoaMUSIC, DoaORMIA, DoaORMIA_MUSIC

class CricketAgent:
    def __init__(self):
        self.mate = False
        self.auditory_sense = None

    def sense(self, position, room_dim, sound_sources, signal, snr):
        self.auditory_sense = DoaORMIA_MUSIC(room_dim, sound_sources, position, snr)
        self.auditory_sense.get_room(signal)
        pred = self.auditory_sense.get_prediction()
        #pred_degree = max(set(pred), key=pred.count)
        return math.pi - pred * math.pi / 180

    def move(self, location, room_dim, sound_sources, signal, snr):
        direction = self.sense(location, room_dim, sound_sources, signal, snr)
        x_align = location[0] + 0.08 * np.cos(direction)
        y_align = location[1] + 0.08 * np.sin(direction)
        return [x_align, y_align, 0]
