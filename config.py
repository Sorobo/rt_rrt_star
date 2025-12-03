# config.py

import numpy as np

# Sampling parameters
ALPHA = 0.4    # probability of sampling along line to goal
BETA = 1         # controls mix of uniform vs ellipse sampling

# Tree density
K_MAX = 30         # max neighbors for rewiring
R_S = 0.3         # minimum spacing between nodes
STEP_SIZE = 0.5  # maximum distance for extending the tree

# Environment/obstacles
GOAL_RADIUS = 1
OBSTACLE_BLOCK_RADIUS = 1.5
OBSTACLE_SENSE_RADIUS = 10.0

# Expansion time budget (seconds)
EXPANSION_BUDGET = 0.050

REWIRE_BUDGET = 0.025

# Grid spatial index resolution
GRID_SIZE = 2.0

# k-step planning horizon
K_PLANNING = 100

# Boat parameters
BOAT_WIDTH = 2.8
BOAT_LENGTH = 5
boat_minimum_dimensium = min(BOAT_WIDTH, BOAT_LENGTH)

# Dynamic planning parameters
USE_DYNAMIC_STEERING = True  # Set to True to use dynamic steering with 6D states
DYNAMIC_DT = 0.1  # Time step for dynamic simulation [s]
DYNAMIC_HORIZON = 10  # Number of steps to simulate ahead
#World bounds
WORLD_BOUNDS = np.array([[0, 10],
                         [0, 10],
                         [-np.pi, np.pi],
                         [-1, 1],
                         [-1, 1],
                         [-1, 1]])  # x, y, theta, vx, vy, omega
SAMPLEBOUNDS = WORLD_BOUNDS.copy()
SAMPLEBOUNDS[0][0]+= boat_minimum_dimensium/2
SAMPLEBOUNDS[0][1]-= boat_minimum_dimensium/2
SAMPLEBOUNDS[1][0]+= boat_minimum_dimensium/2
SAMPLEBOUNDS[1][1]-= boat_minimum_dimensium/2

WORLD_AREA = (WORLD_BOUNDS[0,1] - WORLD_BOUNDS[0,0]) * (WORLD_BOUNDS[1,1] - WORLD_BOUNDS[1,0])* (WORLD_BOUNDS[2,1] - WORLD_BOUNDS[2,0])


