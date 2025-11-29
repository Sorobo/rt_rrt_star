# config.py

import numpy as np

# Sampling parameters
ALPHA = 0.4    # probability of sampling along line to goal
BETA = 1         # controls mix of uniform vs ellipse sampling

# Tree density
K_MAX = 15         # max neighbors for rewiring
R_S = 0.3          # minimum spacing between nodes
STEP_SIZE = 1  # maximum distance for extending the tree

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

#World bounds
WORLD_BOUNDS = np.array([[0, 100],
                         [0, 100]])
WORLD_AREA = (WORLD_BOUNDS[0,1] - WORLD_BOUNDS[0,0]) * (WORLD_BOUNDS[1,1] - WORLD_BOUNDS[1,0])

BOAT_WIDTH = 2.8
BOAT_LENGTH = 5
