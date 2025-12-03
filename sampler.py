# sampler.py

import numpy as np
import random

from config import ALPHA, BETA
from collision import line_collision_free

def sample_uniform(bounds):
    """
    Sample uniformly in configuration space.
    
    Parameters
    ----------
    bounds : np.array
        Bounds for x, y
    include_velocity : bool
        If True, sample 6D state with velocities, else 2D or 3D
    
    Returns
    -------
    np.array
        Sampled state
    """
    # Sample position
    state = np.random.uniform(bounds[:,0], bounds[:,1])
    
   
    return state

def sample_line_to_goal(x0, x_goal):
    t = np.random.uniform()
    return x0 + t * (x_goal - x0)

def sample_ellipse(x0, x_goal, c_best,bounds):
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
    random_part = np.random.uniform(bounds[2:,0], bounds[2:,1])
    pos =(x0 + x_goal)/2 + R @ local
    pos = np.append(pos, random_part)
    return pos


def sample_ellipse_heuristic(x0, x_goal,bounds, expansion_factor=1.5):
    """
    Sample from an ellipse without requiring a path to goal.
    Uses heuristic distance estimate (straight-line distance * expansion_factor).
    
    Parameters
    ----------
    x0 : np.array
        Start position (root of tree)
    x_goal : np.array
        Goal position
    expansion_factor : float
        Multiplier for straight-line distance to create ellipse
        (e.g., 1.5 means ellipse is 50% larger than minimum)
    
    Returns
    -------
    np.array
        Sampled configuration [x, y, theta]
    """
    c_min = np.linalg.norm(x_goal[:2] - x0[:2])
    
    # Use heuristic: assume path will be expansion_factor times the straight-line distance
    c_heuristic = c_min * expansion_factor
    
    # Ensure we have a valid ellipse (c_heuristic > c_min)
    if c_heuristic <= c_min:
        c_heuristic = c_min + 1e-6
    
    # Ellipse major/minor axes
    a = c_heuristic / 2.0
    b = np.sqrt(c_heuristic**2 - c_min**2) / 2.0
    
    # Sample in ellipse coordinate frame (uniform in ellipse)
    r = np.sqrt(np.random.uniform())
    phi = np.random.uniform(0, 2*np.pi)
    local = np.array([r*np.cos(phi)*a, r*np.sin(phi)*b])
    
    # Rotation matrix to align ellipse with start-goal line
    if c_min > 1e-10:
        dir_vec = (x_goal[:2] - x0[:2]) / c_min
        R = np.array([[dir_vec[0], -dir_vec[1]],
                      [dir_vec[1],  dir_vec[0]]])
    else:
        R = np.eye(2)
    
    # Transform to world frame
    center = (x0[:2] + x_goal[:2]) / 2
    pos = center + R @ local
    
    # Random heading
    random_part = np.random.uniform(bounds[2:,0], bounds[2:,1])
    
    return np.append(pos, random_part)

def sample(bounds, x0, x_goal, c_best, path_exists):
    """
    Implements informed sampling strategy.
    
    Sampling probabilities:
    - ALPHA: probability of sampling along line to goal
    - (1-ALPHA)/BETA: probability of uniform sampling
    - Remaining: ellipse sampling (heuristic if no path, informed if path exists)
    """
    Pr = random.random()

    # Sample along line to goal
    if Pr > 1 - ALPHA:
        return sample_line_to_goal(x0, x_goal)

    # Uniform sampling
    if Pr <= (1 - ALPHA) / BETA:
        return sample_uniform(bounds)

    # Ellipse sampling
    if path_exists:
        # Use informed ellipse with actual path cost
        return sample_ellipse(x0, x_goal, c_best,bounds)
    else:
        # Use heuristic ellipse even without a path
        return sample_ellipse_heuristic(x0, x_goal,bounds, expansion_factor=1.5)