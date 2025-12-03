# rewiring.py

import numpy as np
from collections import deque
from collision import line_collision_free, boat_collision_free
from config import K_MAX,REWIRE_BUDGET
from tree import Tree, Node
import time


def dynamic_cost_heuristic(node_from, node_to):
    """
    Approximate dynamic cost considering heading and velocity changes.
    This is a heuristic that penalizes dynamically infeasible connections
    without doing full simulation.
    
    Returns total cost including:
    - Euclidean distance (base cost)
    - Heading change penalty (harder to achieve large heading changes)
    - Velocity change penalty (acceleration/deceleration costs)
    """
    # Base geometric distance
    distance = np.linalg.norm(node_to.x[:2] - node_from.x[:2])
    
    # Heading change penalty
    if len(node_from.x) >= 3 and len(node_to.x) >= 3:
        # Calculate heading difference (wrapped to [-pi, pi])
        heading_diff = np.arctan2(
            np.sin(node_to.x[2] - node_from.x[2]),
            np.cos(node_to.x[2] - node_from.x[2])
        )
        # Penalize large heading changes (boat can't turn instantly)
        # Weight this based on distance - longer distances allow more heading change
        heading_penalty = abs(heading_diff) * 2.0
        
        # Direction consistency: penalize if heading doesn't point toward target
        direction_to_target = np.arctan2(
            node_to.x[1] - node_from.x[1],
            node_to.x[0] - node_from.x[0]
        )
        heading_error = np.arctan2(
            np.sin(node_from.x[2] - direction_to_target),
            np.cos(node_from.x[2] - direction_to_target)
        )
        direction_penalty = abs(heading_error) * 1.0
    else:
        heading_penalty = 0.0
        direction_penalty = 0.0
    
    # Velocity change penalty (if 6D states available)
    if len(node_from.x) >= 6 and len(node_to.x) >= 6:
        # Penalize large velocity changes (requires high forces)
        vel_change = np.linalg.norm(node_to.x[3:6] - node_from.x[3:6])
        velocity_penalty = vel_change * 0.5
    else:
        velocity_penalty = 0.0
    
    # Total cost
    total_cost = distance + heading_penalty + direction_penalty + velocity_penalty
    
    return total_cost

def try_rewire(tree, node, neighbors, obstacles, queue=None):
    """Attempt to make node the parent of each neighbor."""
    for n in neighbors:
        if n is tree.root or n.ineligible:
            continue

        # Use dynamic cost heuristic instead of simple distance
        new_cost = node.cost + dynamic_cost_heuristic(node, n)
        if new_cost < n.cost:
            if boat_collision_free(node, n, obstacles):
                # rewire
                if n.parent:
                    n.parent.children.remove(n)
                node.add_child(n)
                n.cost = new_cost
                if queue is not None:
                    queue.append(n)

def random_rewire(tree, queue, obstacles):
    while queue:
        n_r = queue.pop()
        if n_r.ineligible:
            continue
        neighbors = tree.nearby(n_r.x)
        for near in neighbors:
            if near.ineligible:
                continue
            c_old = tree.cost(near)
            # Use dynamic cost heuristic
            c_new = tree.cost(n_r) + dynamic_cost_heuristic(n_r, near)
            if c_new < c_old and boat_collision_free(n_r, near, obstacles):
                if near.parent:
                    near.parent.children.remove(near)
                n_r.add_child(near)
                near.cost = c_new
                near.parent = n_r


def root_rewire(tree, queue, obstacles,path,k=10):
    """Propagate improved costs outward from the current root."""
    if len(queue) == 0:
        queue.append(path[0])
        
    start_time = time.perf_counter()
    while queue and time.perf_counter() - start_time < REWIRE_BUDGET:
        n_s = queue.pop(0)
        if n_s.ineligible:
            continue
        near_nodes = tree.nearby(n_s.x)
        for near in near_nodes:
            if near.ineligible:
                continue
            c_old = tree.cost(near)
            # Use dynamic cost heuristic
            c_new = tree.cost(n_s) + dynamic_cost_heuristic(n_s, near)
            if c_new < c_old and boat_collision_free(n_s, near, obstacles):
                if near.parent:
                    near.parent.children.remove(near)
                n_s.add_child(near)
                near.cost = c_new
                near.parent = n_s
                queue.append(near)

    #queue.clear()