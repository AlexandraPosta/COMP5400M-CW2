"""
	COMP5400M - CW2
    Author Name: Alexandra Posta - el19a2p
                 Alexandre Monk - el19a2m
                 Bogdan-Alexandru Ciurea - sc20bac
"""

from typing import List
from .speaker import Speaker
from .agent import CricketAgentEvolution


class CricketEnvironment:
    def __init__(self, room_dim: List[float]):
        """
        Initialise the environment

        Args:
            room_dim (List[float]): The dimensions of the room
        """

        self.room_dim: List[float] = room_dim
        self.sources: List[Speaker] = []
        self.agents = []

    def get_room_dimensions(self) -> List[float]:
        """
        Get the dimensions of the room

        Returns:
            List[float]: The dimensions of the room
        """

        return self.room_dim

    def add_agent(self, agent) -> None:
        """
        Add an agent to the environment

        Args:
            agent (CricketAgent): The agent to add
        """

        self.agents.append(agent)

    def get_agent_locations(self) -> List[List[float]]:
        """
        Get the locations of the agents

        Returns:
            List[List[float]]: The locations of the agents
        """

        return [agent.get_position() for agent in self.agents]

    def add_source(self, source: Speaker) -> None:
        """
        Add a sound source to the environment

        Args:
            source (Speaker): The sound source to add
        """

        self.sources.append(source)

    def get_source_locations(self) -> List[List[float]]:
        """
        Get the locations of the sound sources

        Returns:
            List[List[float]]: The locations of the sound sources
        """

        return [source.get_position() for source in self.sources]

    def has_winner(self) -> bool:
        """
        Check if there is a winner

        Returns:
            bool: True if there is a winner, False otherwise
        """

        return any([agent.mate for agent in self.agents])
    
    def get_winner(self) -> CricketAgentEvolution:
        """
        Get the winner

        Returns:
            CricketAgent: The winner
        """

        return [agent for agent in self.agents if agent.mate][0]
    
    def remove_looser_agents(self, count = 1) -> None:
        """
        Remove the looser agents

        Args:
            count (int): The number of agents to remove
        """

        for agent in self.agents:
            if not agent.mate:
                self.agents.remove(agent)
                count -= 1    
                if count == 0:
                    break

    def scramble_agents(self) -> None:
        """
        Scramble the agents
        """

        for agent in self.agents:
            agent.move_to_random_position()
            