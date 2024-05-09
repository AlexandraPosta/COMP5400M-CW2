"""
	COMP5400M - CW2
    Author Name: Alexandra Posta - el19a2p
                 Alexandre Monk - el19a2m
                 Bogdan-Alexandru Ciurea - sc20bac
"""

import math
import random
from typing import List
import numpy as np


class Speaker:
    def __init__(
        self,
        position: List[float] = [
            random.uniform(0.5, 9.5),
            random.uniform(20 / 3, 9.5),
        ],
    ):
        """
        Initialize the speaker

        Args:
            position ([position_x: float, position_y: float], optional): The initial position of the speaker. 
            Defaults to [random.uniform(0, 10), random.uniform(0, 10)].
        """

        self.position: List[float] = position
        self.direction: float = random.uniform(0, 2 * 3.14159)
        self.speed: float = 0.08

    def move(self) -> List[float]:
        """
        Move the speaker in a random direction

        Returns:
            [position_x: float, position_y: float]: The new position of the speaker
        """

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

    def get_position(self) -> List[float]:
        """
        Get the position of the speaker

        Returns:
            [position_x: float, position_y: float]: The position of the speaker
        """

        return self.position

    def __change_direction(self) -> None:
        """
        Change the direction of the speaker by a random angle
        """

        angle = np.random.uniform(-15, 15)
        angle = np.radians(angle)
        self.direction += angle
