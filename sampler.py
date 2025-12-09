# sampler.py

import numpy as np
import random

from config import ALPHA, BETA
from collision import line_collision_free

def sample_uniform(bounds):
    x = np.random.uniform(bounds[:,0], bounds[:,1])
    return x

def sample_line_to_goal(x0, x_goal):
    t = np.random.uniform()
    return x0 + t * (x_goal - x0)

def sample_ellipse(x0, x_goal, c_best):
    """Direct ellipse sampling from Informed RRT*."""
    c_min = np.linalg.norm(x_goal[:2] - x0[:2])
    if c_best < c_min:
        c_best = c_min + 1e-6

    # Ellipse major/minor axes
    a = c_best / 2.0
    b = np.sqrt(c_best**2 - c_min**2) / 2.0

    # sample in ellipse coordinate frame
    r = np.sqrt(np.random.uniform())
    phi = np.random.uniform(0, 2*np.pi)
    local = np.array([r*np.cos(phi)*a, r*np.sin(phi)*b])

    # rotate
    dir_vec = (x_goal - x0) / c_min
    R = np.array([[dir_vec[0], -dir_vec[1]],
                  [dir_vec[1],  dir_vec[0]]])
    theta = np.random.uniform(-np.pi, np.pi)
    pos =(x0 + x_goal)/2 + R @ local
    pos = np.append(pos, theta)
    return pos

def sample_ellipse_heuristic(x0, x_goal, expansion_factor=1.5):
    """
    Sample from a heuristic ellipse even when no path exists yet.
    Uses expansion_factor * straight_line_distance as the major axis.
    """
    c_min = np.linalg.norm(x_goal[:2] - x0[:2])
    c_best = c_min * expansion_factor
    
    # Ellipse major/minor axes
    a = c_best / 2.0
    b = np.sqrt(c_best**2 - c_min**2) / 2.0
    
    # sample in ellipse coordinate frame
    r = np.sqrt(np.random.uniform())
    phi = np.random.uniform(0, 2*np.pi)
    local = np.array([r*np.cos(phi)*a, r*np.sin(phi)*b])
    
    # rotate
    dir_vec = (x_goal - x0) / c_min
    R = np.array([[dir_vec[0], -dir_vec[1]],
                  [dir_vec[1],  dir_vec[0]]])
    theta = np.random.uniform(-np.pi, np.pi)
    pos = (x0 + x_goal)/2 + R @ local
    pos = np.append(pos, theta)
    return pos

def sample(bounds, x0, x_goal, c_best, path_exists):
    """Implements equation (1) from the paper with heuristic ellipse fallback."""
    Pr = random.random()

    if Pr > 1 - ALPHA:
        return sample_line_to_goal(x0, x_goal)

    if Pr <= (1 - ALPHA) / BETA:
        return sample_uniform(bounds)

    # Use informed ellipse if path exists, otherwise use heuristic ellipse
    if path_exists:
        return sample_ellipse(x0, x_goal, c_best)
    else:
        return sample_ellipse_heuristic(x0, x_goal, expansion_factor=1.5)
