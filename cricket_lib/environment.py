from typing import List

from .speaker import Speaker


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
