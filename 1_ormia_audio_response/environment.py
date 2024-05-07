class CricketEnvironment:
    def __init__(self, room_dim, source_locations):
        self.room_dim = room_dim
        self.source_locations = source_locations

    def get_room_dimensions(self):
        return self.room_dim

    def get_source_locations(self):
        return self.source_locations
