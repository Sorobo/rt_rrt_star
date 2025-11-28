# node.py

import numpy as np

class Node:
    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)
        self.parent = None
        self.children = []
        self.cost = 0.0      # cost-to-reach
        self.blocked = False

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)
