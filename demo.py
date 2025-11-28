# demo.py

import numpy as np
from rt_rrt_star import RTRRTStar

bounds = np.array([[0, 100],
                   [0, 100]])

x_start = np.array([10, 10])
x_goal  = np.array([90, 90])

obstacles = [
    (np.array([50,50]), 10),
    (np.array([30,70]), 8)
]

planner = RTRRTStar(bounds, x_start)

x_agent = x_start.copy()

for step in range(2000):
    path = planner.step(x_agent, x_goal, obstacles)

    if len(path) > 1:
        x_agent = path[1].x   # move one step along planned path

    print(f"Step {step}: agent at {x_agent}, path length={len(path)}")
    if step % 100 == 0:
        planner.tree.plot_tree()
    print(len(planner.tree.nodes), "nodes in tree")
