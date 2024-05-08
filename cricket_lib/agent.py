import math
from typing import List
import numpy as np

from doa import DoaCNN


class CricketAgent:

    def __init__(
        self,
        position: List[float] = None,
        speed: float = 1.0,
    ):
        """
        Initialize the agent

        Args:
            position ([position_x: float, position_y: float, position_z: float], optional): The initial position of the agent. Defaults to [np.random.uniform(0, 10), np.random.uniform(0, 10), 0].
            speed (float, optional): The speed of the agent. Defaults to 1.0.
        """

        print("Agent position: ", position)

        self.mate: bool = False
        self.auditory_sense: DoaCNN = None
        self.distance_mic: float = 0.1
        self.snr: float = 0.1
        self.speed: float = speed
        self.position: List[float] = position if position else self.__random_position()

    def sense(
        self,
        position: List[float],
        room_dim: List[float],
        sound_sources: List[List[float]],
        signal: np.array,
    ) -> float:
        """
        Sense the direction of the sound source

        Args:
            position ([position_x: float, position_y: float, position_z: float]): The position of the agent
            room_dim (List[float]): The dimensions of the room
            sound_sources (List[List[float]]): The locations of the sound sources
            signal (np.array): The audio signal

        Returns:
            float: The direction of the sound source
        """

        self.auditory_sense = DoaCNN(
            room_dim, sound_sources, position, self.distance_mic, self.snr
        )
        self.auditory_sense.get_room(signal)
        pred = self.auditory_sense.get_prediction()
        pred_degree = max(set(pred), key=pred.count)
        return math.pi - pred_degree * math.pi / 180

    def move(
        self, room_dim: List[float], sound_sources: List[List[float]], signal: np.array
    ) -> None:
        """
        Move the agent towards the sound source

        Args:
            room_dim (List[float]): The dimensions of the room
            sound_sources (List[List[float]]): The locations of the sound sources
            signal (np.array): The audio signal
        """

        if self.mate:
            return

        direction = self.sense(self.position, room_dim, sound_sources, signal)
        x_align = self.position[0] + 0.08 * np.cos(direction) * self.speed
        y_align = self.position[1] + 0.08 * np.sin(direction) * self.speed

        if x_align < 0:
            x_align = 0
        elif x_align > room_dim[0]:
            x_align = room_dim[0]
        if y_align < 0:
            y_align = 0
        elif y_align > room_dim[1]:
            y_align = room_dim[1]

        self.position = [x_align, y_align, 0]

        self.check_mate(sound_sources)

    def get_position(self) -> list:
        """
        Get the position of the agent

        Returns:
            [position_x: float, position_y: float, position_z: float]: The position of the agent
        """

        return self.position

    def check_mate(self, sound_sources: List[List[float]]) -> bool:
        """
        Check if the agent has reached the sound source

        Args:
            sound_sources (List[List[float]]): The locations of the sound sources

        Returns:
            bool: True if the agent has reached the sound source, False otherwise
        """

        # If the agent is 0.3 units away from any sound source, then the agent has reached the sound source
        for source in sound_sources:
            distance = math.sqrt(
                (source[0] - self.position[0]) ** 2
                + (source[1] - self.position[1]) ** 2
            )
            if distance < 0.3:
                self.mate = True
                return self.mate

        # Or if the agent has reached the top of the room
        if self.position[1] >= 10:
            self.mate = True
            return self.mate

        return self.mate

    def __random_position(self) -> List[float]:
        """
        Generate a random position for the agent

        Returns:
            List[float]: The random position of the agent
        """

        return [np.random.uniform(0.5, 9.5), np.random.uniform(0.5, 20 / 3 - 0.5), 0]


class CricketAgentMemory(CricketAgent):

    def __init__(
        self,
        memory_size: int = 8,
        position=[np.random.uniform(0.5, 9.5), np.random.uniform(0.5, 20 / 3 - 0.5), 0],
        speed: float = 1.0,
    ):
        """
        Initialize the agent with memory

        Args:
            memory_size (int, optional): The size of the memory. Defaults to 8.
        """

        super().__init__(position=position, speed=speed)
        self.angle_memory: List[List[float]] = (
            []
        )  # Memory to store recent angle distributions
        self.memory_size: int = memory_size  # How many recent distributions to remember

    def move(
        self, room_dim: List[float], sound_sources: List[List[float]], signal: np.array
    ) -> None:
        """
        Move the agent towards the sound source using memory

        Args:
            room_dim (List[float]): The dimensions of the room
            sound_sources (List[List[float]]): The locations of the sound sources
            signal (np.array): The audio signal
        """

        if self.mate:
            return

        current_distribution = self.sense(
            self.position, room_dim, sound_sources, signal
        )
        self.angle_memory.append(current_distribution)

        # Maintain memory size
        if len(self.angle_memory) > self.memory_size:
            self.angle_memory.pop(0)

        # Combine distributions from memory to form a single distribution
        combined_distribution = self.combine_distributions(self.angle_memory)

        # Choose the most probable direction based on the combined distribution
        weighted_direction = self.calculate_weighted_direction(combined_distribution)

        x_align = self.position[0] + 0.08 * np.cos(weighted_direction) * self.speed
        y_align = self.position[1] + 0.08 * np.sin(weighted_direction) * self.speed

        if x_align < 0:
            x_align = 0
        elif x_align > room_dim[0]:
            x_align = room_dim[0]
        if y_align < 0:
            y_align = 0
        elif y_align > room_dim[1]:
            y_align = room_dim[1]

        self.position = [x_align, y_align, 0]

        self.check_mate(sound_sources)

    def combine_distributions(self, angle_memory: List[List[float]]) -> List[float]:
        """
        Combine the angle distributions from memory

        Args:
            angle_memory (List[List[float]]): The angle distributions from memory

        Returns:
            List[float]: The combined angle distribution
        """

        combined = np.mean([np.array(d) for d in angle_memory], axis=0)
        return combined

    def calculate_weighted_direction(self, combined_distribution: List[float]) -> float:
        """
        Calculate the weighted direction based on the combined distribution

        Args:
            combined_distribution (List[float]): The combined angle distribution

        Returns:
            float: The weighted direction
        """

        return math.pi - np.argmax(combined_distribution) * math.pi / 180
