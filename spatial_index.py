# spatial_index.py

import math
from collections import defaultdict

from config import GRID_SIZE

class SpatialIndex:
    def __init__(self):
        self.grid = defaultdict(list)

    def _key(self, x):
        return (int(x[0] // GRID_SIZE), int(x[1] // GRID_SIZE))

    def insert(self, node):
        self.grid[self._key(node.x)].append(node)

    def remove(self, node):
        key = self._key(node.x)
        if node in self.grid[key]:
            self.grid[key].remove(node)

    def nearby_nodes(self, x):
        """Return nodes in the 9 neighboring grid cells."""
        base = self._key(x)
        nodes = []
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                key = (base[0] + dx, base[1] + dy)
                nodes.extend(self.grid.get(key, []))
        return nodes
