from speaker import Speaker


class CricketEnvironment:
    def __init__(self, room_dim):
        self.room_dim = room_dim
        self.sources = []

    def get_room_dimensions(self):
        return self.room_dim

    def add_source(self, source: Speaker):
        self.sources.append(source)

    def get_source_locations(self):
        return [source.get_position() for source in self.sources]
