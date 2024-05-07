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
    

class CricketAgentMemory:
    def __init__(self):
        self.mate = False
        self.auditory_sense = None
        self.distance_mic = 0.1
        self.snr = 0.1
        self.angle_memory = []  # Memory to store recent angle distributions
        self.memory_size = 8    # How many recent distributions to remember

    def sense(self, position, room_dim, sound_sources, signal):
        self.auditory_sense = DoaCNN(room_dim, sound_sources, position, self.distance_mic, self.snr)
        self.auditory_sense.get_room(signal)
        pred_distribution = self.auditory_sense.get_prediction()
        return pred_distribution
    
    def move(self, location, room_dim, sound_sources, signal):
        current_distribution = self.sense(location, room_dim, sound_sources, signal)
        self.angle_memory.append(current_distribution)
        
        # Maintain memory size
        if len(self.angle_memory) > self.memory_size:
            self.angle_memory.pop(0)

        # Combine distributions from memory to form a single distribution
        combined_distribution = self.combine_distributions(self.angle_memory)

        # Choose the most probable direction based on the combined distribution
        weighted_direction = self.calculate_weighted_direction(combined_distribution)

        x_align = location[0] + 0.08 * np.cos(weighted_direction)
        y_align = location[1] + 0.08 * np.sin(weighted_direction)
        return [x_align, y_align, 0]

    def combine_distributions(self, distributions):
        # This function averages the angle distributions
        # Each distribution is assumed to be a list of tuples (angle, probability)
        combined = np.mean([np.array(d) for d in distributions], axis=0)
        return combined

    def calculate_weighted_direction(self, distribution):
        # Calculate the direction weighted by probability
        angles, probabilities = zip(*distribution)
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()  # Normalize probabilities
        angles = np.array(angles)
        weighted_sum = np.dot(angles, probabilities)
        return weighted_sum
