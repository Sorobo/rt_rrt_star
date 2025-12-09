"""
Configuration file for A* path planner.
"""

import numpy as np

# Grid parameters
GRID_RESOLUTION = 0.5  # meters per grid cell
HEADING_RESOLUTION = np.pi / 4  # radians (22.5 degrees)

# Planning parameters
MAX_ITERATIONS = 10000
GOAL_TOLERANCE_XY = 1.0  # meters
GOAL_TOLERANCE_HEADING = np.pi / 6  # radians (30 degrees)

# Real-time planning
TIME_BUDGET_PER_STEP = 0.02  # seconds (20ms per planning step for smoother framerate)
ITERATIONS_PER_CHECK = 5  # Check time budget every N iterations

# Cost weights
COST_STEP = 1.0  # Cost per grid cell movement
COST_TURN = 0.5  # Cost per heading change
COST_REVERSE = 0.5  # Extra cost for reversing (reduced to make backing up more viable)

# Heuristic weights
HEURISTIC_WEIGHT = 1.0  # Multiplier for heuristic (1.0 = admissible, >1.0 = weighted A*)
HEADING_WEIGHT = 0.3  # Weight for heading difference in heuristic

# Motion primitives (simplified - arc-based movements)
# Each primitive: (forward_distance, turning_angle)
MOTION_PRIMITIVES = [
    # Forward motions
    (1.0, 0.0),           # Straight ahead
    (1.0, np.pi/8),       # Slight right turn
    (1.0, -np.pi/8),      # Slight left turn
    (1.0, np.pi/4),       # Medium right turn
    (1.0, -np.pi/4),      # Medium left turn
    (1.0, np.pi/2),       # Sharp right turn
    (1.0, -np.pi/2),      # Sharp left turn
    
    # Backward motions (more options)
    (-0.8, 0.0),          # Reverse straight (longer)
    (-0.8, np.pi/8),      # Reverse with slight turn right
    (-0.8, -np.pi/8),     # Reverse with slight turn left
    (-0.5, np.pi/4),      # Reverse with medium turn right
    (-0.5, -np.pi/4),     # Reverse with medium turn left
]

# Convert primitives to grid units
MOTION_PRIMITIVES_GRID = [
    (dist / GRID_RESOLUTION, angle) 
    for dist, angle in MOTION_PRIMITIVES
]

# Collision checking
COLLISION_CHECK_RESOLUTION = 0.1  # meters - interpolation step for swept hull check
OBSTACLE_INFLATION = 0.2  # meters - safety margin around obstacles

# Visualization
SHOW_CLOSED_NODES = True  # Show explored nodes
SHOW_OPEN_NODES = True    # Show nodes in open set
SHOW_CURRENT_NODE = True  # Highlight current node being expanded
MAX_NODES_TO_DRAW = 30000  # Limit nodes drawn for performance (reduced for better FPS)
