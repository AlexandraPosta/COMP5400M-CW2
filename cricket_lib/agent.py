import math
import numpy as np
from typing import List, Dict
from collections import Counter

from .doa import DoaCNN, DoaMUSIC, DoaORMIA_CNN, DoaORMIA_MUSIC, DoaORMIA


class CricketAgent:

    def __init__(
        self,
        position: List[float] = None,
        speed: float = 1.0,
        available_space: List[float] = [10.0, 10.0],
    ):
        """
        Initialize the agent

        Args:
            position ([position_x: float, position_y: float, position_z: float], optional): The initial position of the agent. Defaults to [np.random.uniform(0, 10), np.random.uniform(0, 10), 0].
            speed (float, optional): The speed of the agent. Defaults to 1.0.
        """

        self.mate: bool = False
        self.auditory_sense: DoaCNN = None
        self.distance_mic: float = 0.1
        self.snr: float = 0.1
        self.speed: float = speed
        self.available_space: List[float] = available_space
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

        # If the agent is 0.2 units away from any sound source, then the agent has reached the sound source
        for source in sound_sources:
        if (source[0]-0.2 < self.position[0] < source[0]+0.2 and source[1]-0.2 < self.position[1] < source[1]+0.2):
                self.mate = True
                return self.mate

        # Or if the agent has reached the top of the room
        if self.position[0] < 0.1 or self.position[0] > self.available_space[0] - 0.1:
            self.mate = True
            return self.mate

        if self.position[1] < 0.1 or self.position[1] > self.available_space[1] - 0.1:
            self.mate = True
            return self.mate

        return self.mate

    def __random_position(self) -> List[float]:
        """
        Generate a random position for the agent

        Returns:
            List[float]: The random position of the agent
        """

        return [np.random.uniform(0.5, self.available_space[0]-0.5),
                np.random.uniform(0.5, self.available_space[1]/2-0.5), 0]


class CricketAgentMemory(CricketAgent):

    def __init__(
        self,
        position: List[float] = None,
        speed: float = 1.0,
        available_space: List[float] = [10.0, 10.0],
        memory_size: int = 10,
        decay_rate: float = 0.9,
        learning_rate: float = 0.01,    # Learning rate for adaptation
    ):
        super().__init__(position=position, speed=speed, available_space=available_space)
        self.memory_size = memory_size
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.angle_memory: List[Dict[float, float]] = []    # Storing angle and probabilities
        self.adaptation_factor = 0.01                       # Initial adaptation factor

    def sense(
        self,
        position: List[float],
        room_dim: List[float],
        sound_sources: List[List[float]],
        signal: np.array,
    ) -> float:

        self.auditory_sense = DoaCNN(
            room_dim, sound_sources, position, self.distance_mic, self.snr
        )
        self.auditory_sense.get_room(signal)
        angles = self.auditory_sense.get_prediction()
        angle_counts = Counter(angles)  # Count occurrence of each angle
        total_angles = len(angles)

        # Create a dictionary of all angles from 0 to 180 with zero probability
        angle_probabilities = {angle: 0 for angle in range(0, 181, 10)}

        # Update the dictionary with actual probabilities from predictions
        for angle, count in angle_counts.items():
            if angle in angle_probabilities:
                angle_probabilities[angle] = count / total_angles
        return angle_probabilities

    def move(
        self, room_dim: List[float], sound_sources: List[List[float]], signal: np.array
    ) -> None:
            
        if self.mate:
            return

        current_probabilities = self.sense(self.position, room_dim, sound_sources, signal)
        self.angle_memory.append(current_probabilities)
        self.angle_memory = self.angle_memory[-self.memory_size:]

        # Calculate the new direction with weighted average
        weighted_direction = self.calculate_weighted_direction()

        # Update adaptation factor based on recent movement success
        self.update_adaptation_factor(weighted_direction)

        direction = math.pi - weighted_direction * math.pi / 180
        x_align = self.position[0] + self.adaptation_factor * np.cos(direction) * self.speed
        y_align = self.position[1] + self.adaptation_factor * np.sin(direction) * self.speed

        if x_align < 0:
            x_align = 0
        if y_align < 0:
            y_align = 0
        elif y_align > room_dim[1]:
            y_align = room_dim[1]

        self.position = [x_align, y_align, 0]

        self.check_mate(sound_sources)

    def calculate_weighted_direction(self) -> float:
        # Initialize variables to store the sum of weighted probabilities and the total weight for normalization
        weighted_sum = 0
        total_weight = 0

        # Iterate over each memory entry
        for i, memory in enumerate(reversed(self.angle_memory)):
            decay_factor = self.decay_rate ** (i * 2)
            # Filter and apply decay to probabilities above the threshold
            for angle, probability in memory.items():
                if probability >= 0.09:  # Apply the threshold filter
                    adjusted_probability = probability * decay_factor
                    weighted_sum += angle * adjusted_probability
                    total_weight += adjusted_probability

        # Check for total weight to avoid division by zero
        if total_weight == 0:
            return 0  # No valid data to determine direction

        # Calculate the weighted average of the angles
        return weighted_sum / total_weight
    
    def update_adaptation_factor(self, weighted_direction):
        # Adapt based on the direction being positive or negative
        if weighted_direction > 0:
            self.adaptation_factor += self.learning_rate
        else:
            self.adaptation_factor -= self.learning_rate
        # Ensure adaptation factor stays within reasonable limits
        self.adaptation_factor = min(max(self.adaptation_factor, 0.01), 0.5)
