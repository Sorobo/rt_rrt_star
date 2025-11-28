# collision.py

import numpy as np

def line_collision_free(x1, x2, obstacles):
    """Very simple checker: return True if straight line is collision-free."""
    for obs in obstacles:
        center, radius = obs
        # Distance from segment to obstacle center
        v = x2 - x1
        w = center - x1
        t = max(0, min(1, np.dot(v, w) / np.dot(v, v)))
        closest = x1 + t * v
        if np.linalg.norm(closest - center) < radius:
            return False
    return True
