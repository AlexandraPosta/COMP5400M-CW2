import math
import random
import numpy as np

class Speaker:
    def __init__(
        self, position: tuple = (random.uniform(0, 10), random.uniform(0, 10))
    ):
        self.position = position
        self.direction = random.uniform(0, 2 * 3.14159)
        self.speed = 0.08

    def move(self):
        # Move the speaker
        self.position = (
            self.position[0] + self.speed * math.cos(self.direction),
            self.position[1] + self.speed * math.sin(self.direction),
        )

        # Check if the speaker is out of bounds
        if self.position[0] < 0:
            self.position = (0, self.position[1])
        elif self.position[0] > 10:
            self.position = (10, self.position[1])

        if self.position[1] < 0:
            self.position = (self.position[0], 0)
        elif self.position[1] > 10:
            self.position = (self.position[0], 10)

        self.__change_direction()
        return self.position

    def __change_direction(self):
        angle = np.random.uniform(-15, 15)
        angle = np.radians(angle)
        self.direction += angle

    def get_position(self):
        return self.position
