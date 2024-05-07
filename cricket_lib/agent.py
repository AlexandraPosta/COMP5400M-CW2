import math
import numpy as np
from doa import DoaCNN

class CricketAgent:
    def __init__(
        self,
        position=[np.random.uniform(0, 10), np.random.uniform(0, 10), 0],
        speed: float = 1.0,
    ):
        self.mate: bool = False
        self.auditory_sense = None
        self.distance_mic: float = 0.1
        self.snr: float = 0.1
        self.speed: float = speed
        self.position: list = position

    def sense(self, position, room_dim, sound_sources, signal) -> float:
        self.auditory_sense = DoaCNN(room_dim, sound_sources, position, self.distance_mic, self.snr)
        self.auditory_sense.get_room(signal)
        pred = self.auditory_sense.get_prediction()
        pred_degree = max(set(pred), key=pred.count)
        return math.pi - pred_degree * math.pi / 180

    def move(self, room_dim, sound_sources, signal) -> None:
        if self.mate:
            return
        direction = self.sense(self.position, room_dim, sound_sources, signal)
        x_align = self.position[0] + 0.08 * np.cos(direction) * self.speed
        y_align = self.position[1] + 0.08 * np.sin(direction) * self.speed
        self.position = [x_align, y_align, 0]
        return self.position

    def get_position(self) -> list:
        return self.position

    def check_mate(self, sound_sources: list) -> bool:
        # If the agent is 0.3 units away from any sound source, then the agent has reached the sound source
        for source in sound_sources:
            distance = math.sqrt(
                (source[0] - self.position[0]) ** 2
                + (source[1] - self.position[1]) ** 2
            )
            if distance < 0.3:
                self.mate = True
                return self.mate

        return self.mate
