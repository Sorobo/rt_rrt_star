# RT-RRT* Path Planner

Real-time RRT* (Rapidly-exploring Random Tree Star) path planning algorithm with dynamic obstacle avoidance.

## Features

- Real-time path planning with RT-RRT* algorithm
- Dynamic obstacle support with automatic node blocking/unblocking
- Pygame-based visualization
- Spatial indexing using R-tree for efficient nearest neighbor queries
- Interactive goal setting and obstacle placement

## Requirements

- Python 3.x
- numpy
- pygame
- rtree

## Installation

```bash
pip install numpy pygame rtree
```

## Usage

Run the interactive demo:

```bash
python pygama_demo.py
```

- **Left-click**: Set new goal position
- **Right-click**: Add static obstacle

## Project Structure

- `rt_rrt_star.py` - Main RT-RRT* algorithm implementation
- `tree.py` - Tree data structure with spatial indexing
- `node_module.py` - Node class definition
- `planner.py` - Path planning utilities
- `sampler.py` - Sampling strategies
- `rewiring.py` - Tree rewiring operations
- `collision.py` - Collision detection
- `dynamic_obstacle.py` - Dynamic obstacle class
- `rtree_module.py` - R-tree spatial index
- `pygama_demo.py` - Pygame visualization
- `config.py` - Configuration parameters

## Configuration

Edit `config.py` to adjust:
- Expansion budget
- Tree density parameters
- Obstacle blocking radius
- World bounds
