# node.py

import numpy as np

class Node:
    def __init__(self, x):
        """
        Initialize a node with state vector.
        
        Parameters
        ----------
        x : array-like
            State vector. Can be:
            - [x, y] for 2D position only
            - [x, y, theta] for 2D position + heading
            - [x, y, theta, x_dot, y_dot, theta_dot] for full 6D state
        """
        self.x = np.asarray(x, dtype=float)
        self.parent = None
        self.children = []
        self.cost = 0.0      # cost-to-reach
        self.blocked = False
        self.ineligible = False  # True if all dynamic trajectories lead to collision
        
        # Store control input that led to this state (for dynamic steering)
        self.control = None
    
    @property
    def position(self):
        """Get position [x, y]"""
        return self.x[:2]
    
    @property
    def heading(self):
        """Get heading angle (theta)"""
        if len(self.x) >= 3:
            return self.x[2]
        return 0.0
    
    @property
    def velocity(self):
        """Get velocity [x_dot, y_dot, theta_dot]"""
        if len(self.x) >= 6:
            return self.x[3:6]
        return np.zeros(3)

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)
