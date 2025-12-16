# config.py

import numpy as np

# Sampling parameters
ALPHA = 0.1    # probability of sampling along line to goal
BETA = 1         # controls mix of uniform vs ellipse sampling hihj more ellipse low more uniform

# Tree density
K_MAX = 5         # max neighbors for rewiring
R_S = 1          # minimum spacing between nodes
STEP_SIZE = 2.5  # maximum distance for extending the tree

# Environment/obstacles
GOAL_RADIUS = 1
OBSTACLE_BLOCK_RADIUS = 2
OBSTACLE_SENSE_RADIUS = 10.0

# Expansion time budget (seconds)
EXPANSION_BUDGET = 0.050

REWIRE_BUDGET = 0.05

# Grid spatial index resolution
GRID_SIZE = 2.0

# k-step planning horizon
K_PLANNING = 100
BOAT_WIDTH = 2.8
BOAT_LENGTH = 5
BOAT_SAFETY_PADDING = 0.5  # Extra padding around boat for collision detection
boat_minimum_dimensium = min(BOAT_WIDTH, BOAT_LENGTH)
#World bounds
WORLD_BOUNDS = np.array([[0, 30],
                         [0, 30],
                         [-np.pi, np.pi]])  # x, y, theta
SAMPLEBOUNDS = WORLD_BOUNDS.copy()
SAMPLEBOUNDS[0][0]+= boat_minimum_dimensium/2
SAMPLEBOUNDS[0][1]-= boat_minimum_dimensium/2
SAMPLEBOUNDS[1][0]+= boat_minimum_dimensium/2
SAMPLEBOUNDS[1][1]-= boat_minimum_dimensium/2

WORLD_AREA = (WORLD_BOUNDS[0,1] - WORLD_BOUNDS[0,0]) * (WORLD_BOUNDS[1,1] - WORLD_BOUNDS[1,0])* (WORLD_BOUNDS[2,1] - WORLD_BOUNDS[2,0])
