# rewiring.py

import numpy as np
from collections import deque
from collision import line_collision_free, boat_collision_free
from config import K_MAX,REWIRE_BUDGET
from tree import Tree, Node
import time

def try_rewire(tree, node, neighbors, obstacles, queue=None):
    """Attempt to make node the parent of each neighbor."""
    for n in neighbors:
        if n is tree.root:
            continue

        new_cost = node.cost + np.linalg.norm(n.x - node.x)
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
        neighbors = tree.nearby(n_r.x)
        for near in neighbors:
            c_old = tree.cost(near)
            c_new = tree.cost(n_r) + np.linalg.norm(near.x - n_r.x)
            if c_new < c_old and boat_collision_free(n_r, near, obstacles):
                if near.parent:
                    near.parent.children.remove(near)
                n_r.add_child(near)
                near.cost = c_new
                near.parent = n_r
                # Update heading based on new parent
                direction = near.x - n_r.x
                if np.linalg.norm(direction) > 1e-6:
                    near.heading = np.arctan2(direction[1], direction[0])


def root_rewire(tree, queue, obstacles,path,k=10):
    """Propagate improved costs outward from the current root."""
    if len(queue) == 0:
        queue.append(path[0])
        
    start_time = time.perf_counter()
    while queue and time.perf_counter() - start_time < REWIRE_BUDGET:
        n_s = queue.pop(0)
        near_nodes = tree.nearby(n_s.x)
        #print("search area is", len(near_nodes),"with queue length", len(queue))
        for near in near_nodes:
            c_old = tree.cost(near)
            c_new = tree.cost(n_s) + np.linalg.norm(near.x - n_s.x)
            if c_new < c_old and boat_collision_free(n_s, near, obstacles):
                if near.parent:
                    near.parent.children.remove(near)
                n_s.add_child(near)
                near.cost = c_new
                near.parent = n_s
                # Update heading based on new parent
                direction = near.x - n_s.x
                if np.linalg.norm(direction) > 1e-6:
                    near.heading = np.arctan2(direction[1], direction[0])
                queue.append(near)

    #queue.clear()