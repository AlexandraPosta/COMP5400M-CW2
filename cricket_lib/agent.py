"""
	COMP5400M - CW2
    Author Name: Alexandra Posta - el19a2p
                 Alexandre Monk - el19a2m
                 Bogdan-Alexandru Ciurea - sc20bac
"""

import math
import numpy as np
from typing import List, Dict
from collections import Counter
from .doa import DoaCNN, DoaMUSIC, DoaORMIA_CNN, DoaORMIA_MUSIC, DoaORMIA


class CricketAgent:
    """
        Initialize the insect agent

        Args:
            position ([position_x: float, position_y: float, position_z: float], optional): The initial position of the agent. Defaults to [np.random.uniform(0, 10), np.random.uniform(0, 10), 0].
            speed (float, optional): The speed of the agent. Defaults to 1.0.
            available_space (List[float], optional): The available space in the room. Defaults to [10.0, 10.0].
    """

    def __init__(
        self,
        position: List[float] = None,
        speed: float = 1.0,
        available_space: List[float] = [10.0, 10.0],
    ):
        self.mate: bool = False
        self.auditory_sense: DoaCNN = None
        self.distance_mic: float = 0.1
        self.snr: float = 0.1
        self.speed: float = speed
        self.available_space: List[float] = available_space
        self.position: List[float] = position if position else self.random_position()
        self.past_positions: List[List[float]] = [] # Used for visualisation

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
            if (
                source[0]-0.2 < self.position[0] < source[0]+0.2 and 
                source[1]-0.2 < self.position[1] < source[1]+0.2
                ):
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

    def random_position(self) -> List[float]:
        """
        Generate a random position for the agent

        Returns:
            List[float]: The random position of the agent
        """

        return [
            np.random.uniform(0.5, self.available_space[0] - 0.5),
            np.random.uniform(0.5, self.available_space[1] / 2 - 0.5),
            0,
        ]


class CricketAgentEvolution(CricketAgent):
    """
    Initialise the insect agent with evolution

    Args:
        position ([position_x: float, position_y: float, position_z: float], optional): The initial position of the agent. 
        Defaults to [np.random.uniform(0, 10), np.random.uniform(0, 10), 0].
        speed (float, optional): The speed of the agent. Defaults to 1.0.
        available_space (List[float], optional): The available space in the room. Defaults to [10.0, 10.0].
        mutation_rate (float, optional): The mutation rate for evolution. Defaults to 0.1.
    """

    def __init__(
        self,
        position: List[float] = None,
        speed: float = 1.0,
        available_space: List[float] = [10.0, 10.0],
        mutation_rate: float = 0.1,
    ):
        super().__init__(
            position=position, speed=speed, available_space=available_space
        )
        self.mutation_rate = mutation_rate
        self.color = "red"

    def get_fitness(self) -> float:
        """
        Get the fitness of the agent
        Returns:
            float: The fitness of the agent
        """

        traversed_distance = 0
        for i in range(1, len(self.past_positions)):
            traversed_distance += math.sqrt(
                (self.past_positions[i][0] - self.past_positions[i - 1][0]) ** 2
                + (self.past_positions[i][1] - self.past_positions[i - 1][1]) ** 2
            )

        return 1 / traversed_distance if traversed_distance != 0 else 0

    def evolve(self) -> None:
        """
        Evolve the agent
        
        Changes just it's speed
        
        """

        self.speed += np.random.uniform(-self.mutation_rate, self.mutation_rate)

    def move_to_random_position(self) -> None:
        """
        Move the agent to a random position
        """

        self.position = self.random_position()


class CricketAgentMemory(CricketAgent):
    """
    Initialise the insect agent with memory

    Args:
        position ([position_x: float, position_y: float, position_z: float], optional): The initial position of the agent. 
        Defaults to [np.random.uniform(0, 10), np.random.uniform(0, 10), 0].
        speed (float, optional): The speed of the agent. Defaults to 1.0.
        available_space (List[float], optional): The available space in the room. Defaults to [10.0, 10.0].
        memory_size (int, optional): The size of the memory. Defaults to 10.
        decay_rate (float, optional): The decay rate for memory. Defaults to 0.9.
        confidence_rate (float, optional): The confidence rate for adaptation. Defaults to 0.01.
    """

    def __init__(
        self,
        position: List[float] = None,
        speed: float = 1.0,
        available_space: List[float] = [10.0, 10.0],
        memory_size: int = 10,
        decay_rate: float = 0.9,
        confidence_rate: float = 0.01,    # Confidence for adaptation
    ):
        super().__init__(position=position, speed=speed, available_space=available_space)
        self.memory_size = memory_size
        self.decay_rate = decay_rate
        self.confidence_rate = confidence_rate
        self.angle_memory: List[Dict[float, float]] = []    # Storing angle and probabilities
        self.confidence_factor = 0.03                       # Initial adaptation factor

    def sense_distribution(
        self,
        position: List[float],
        room_dim: List[float],
        sound_sources: List[List[float]],
        signal: np.array,
    ) -> float:
        """
        Sense the distribution of the sound source

        Args:
            position ([position_x: float, position_y: float, position_z: float]): The position of the agent
            room_dim (List[float]): The dimensions of the room
            sound_sources (List[List[float]]): The locations of the sound sources
            signal (np.array): The audio signal

        Returns:
            Probability distrobution as dictionary of {incoming angle: probability of incoming angle}
        """

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
        """
            Move the agent towards the sound source using memory

            Args:
                room_dim (List[float]): The dimensions of the room
                sound_sources (List[List[float]]): The locations of the sound sources
                signal (np.array): The audio signal
        """
            
        if self.mate:
            return

        if self.position[1] < room_dim[1]/2:
            current_probabilities = self.sense_distribution(self.position, room_dim, sound_sources, signal)
            self.angle_memory.append(current_probabilities)
            self.angle_memory = self.angle_memory[-self.memory_size:]
            # Calculate the new direction with weighted average
            weighted_direction = self.calculate_weighted_direction()
            direction = math.pi - weighted_direction * math.pi / 180
        else:
            direction = self.sense(self.position, room_dim, sound_sources, signal)

        # Update adaptation factor based on recent movement success
        self.update_confidence_factor(direction)
        
        x_align = self.position[0] + self.confidence_factor * np.cos(direction) * self.speed
        y_align = self.position[1] + self.confidence_factor * np.sin(direction) * self.speed
        self.position = [x_align, y_align, 0]
        self.check_mate(sound_sources)

    def calculate_weighted_direction(self) -> float:
        """
        Calculate the weighted direction based on the memory

        Returns:
            float: The weighted direction of arrival of sound
        """

        # Initialize variables to store the sum of weighted probabilities and the total weight for normalization
        weighted_sum = 0
        total_weight = 0

        # Iterate over each memory entry
        for i, memory in enumerate(reversed(self.angle_memory)):
            decay_factor = self.decay_rate ** (i * 2)
            # Filter and apply decay to probabilities above the threshold
            for angle, probability in memory.items():
                if probability >= 0.08:  # Apply the threshold filter
                    adjusted_probability = probability * decay_factor
                    weighted_sum += angle * adjusted_probability
                    total_weight += adjusted_probability

        # Check for total weight to avoid division by zero
        if total_weight == 0:
            return 0  # No valid data to determine direction

        # Calculate the weighted average of the angles
        return weighted_sum / total_weight
    
    def update_confidence_factor(self, direction):
        """
        Update the confidence factor based on the direction of movement. A straight line stimulates the
        movement or confidence factor, while a sharp turn decreases the confidence factor.

        Args:
            direction (float): The direction of movement
        """
        if 1.4 < direction < 1.75:
            self.confidence_factor += self.confidence_rate  # Increase if nearly straight ahead
        elif direction < 0.7 or direction > 2.45:
            self.confidence_factor -= self.confidence_rate

        # Ensure adaptation factor stays within reasonable limits
        self.confidence_factor = min(max(self.confidence_factor, 0.03), 0.2)
