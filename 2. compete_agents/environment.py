from speaker import Speaker
from agent import CricketAgent
from typing import List


class CricketEnvironment:
    def __init__(self, room_dim):
        self.room_dim = room_dim
        self.sources: List[Speaker] = []
        self.agents: List[CricketAgent] = []

    def get_room_dimensions(self) -> List[float]:
        return self.room_dim

    def add_agent(self, agent: CricketAgent) -> None:
        self.agents.append(agent)

    def get_agent_locations(self) -> List[List[float]]:
        return [agent.get_position() for agent in self.agents]

    def add_source(self, source: Speaker) -> None:
        self.sources.append(source)

    def get_source_locations(self) -> List[List[float]]:
        return [source.get_position() for source in self.sources]
