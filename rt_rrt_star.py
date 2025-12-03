# rt_rrt_star.py

import time
import numpy as np
from sampler import sample
from tree import Tree
from rewiring import random_rewire, root_rewire
from planner import plan_k_steps,greedy_path_to_goal,plan_to_goal
from collision import line_collision_free,boat_collision_free
from config import EXPANSION_BUDGET, K_MAX, R_S, OBSTACLE_BLOCK_RADIUS,SAMPLEBOUNDS,WORLD_BOUNDS
from node_module import Node
from steer import steer,steer_dynamic

class RTRRTStar:
    def __init__(self, bounds, x_start, rtree_weights=None):
        self.bounds = bounds
        self.tree = Tree(x_start, rtree_weights=rtree_weights)
        self.c_best = np.inf
        self.path_exists = False
        self.path = [self.tree.root]
        self.Qr = []
        self.Qs = []
        self.step_counter = 0

    def step(self, x_agent, x_goal, obstacles, dynamic_obstacles=None, dt=0.0):

        # Update dynamic obstacles if provided
        if dynamic_obstacles is None:
            dynamic_obstacles = []
        
        for dyn_obs in dynamic_obstacles:
            dyn_obs.update(dt)
            dyn_obs.bounce_in_bounds(self.bounds)
        
        # Combine static and dynamic obstacles for collision checking
        all_obstacles = obstacles + [dyn_obs.get_tuple() for dyn_obs in dynamic_obstacles]
        
        # Block nodes near dynamic obstacles
        for dyn_obs in dynamic_obstacles:
            self.tree.block_nodes_near_obstacle(dyn_obs.center, 
                                               OBSTACLE_BLOCK_RADIUS + dyn_obs.radius)
        
        # Periodically unblock nodes that are clear
        if self.step_counter % 10 == 0:
            self.tree.unblock_nodes(all_obstacles)
        
        self.step_counter += 1
        
        start_time = time.perf_counter()
        
        # ---- Expansion + Rewiring loop (Algorithm 1) ----
        while time.perf_counter() - start_time < EXPANSION_BUDGET:
            # Sample

            x_rand = sample(WORLD_BOUNDS, self.tree.root.x, x_goal,
                            self.c_best, self.path_exists)

            # Nearest
            n_closest = self.tree.nearest_node(x_rand)
            
            # Skip if nearest node is ineligible
            if n_closest.ineligible:
                continue
            
            x_rand, control = steer_dynamic(n_closest.x, x_rand, all_obstacles,horizon = 5)
            
            # Check if steered state is collision-free
            collision_free = boat_collision_free(n_closest, Node(x_rand), all_obstacles)
            
            if collision_free:
                near_nodes = self.tree.nearby(x_rand)
                if len(near_nodes) < K_MAX or np.linalg.norm(n_closest.x - x_rand) > R_S:
                    # Add node
                    new_node = self.tree.add_node(x_rand, n_closest, all_obstacles)
                    self.Qr.insert(0, new_node)
                else:
                    self.Qr.insert(0,n_closest)
            else:
                # Mark node as ineligible if steering always leads to collision
                # This happens when all control options result in collision
                n_closest.collision_attempts = getattr(n_closest, 'collision_attempts', 0) + 1
                if n_closest.collision_attempts >= 5:  # After multiple failures
                    n_closest.ineligible = True
            
            # Rewiring
            random_rewire(self.tree, self.Qr, all_obstacles)
        
        if len(self.path) >= 2:
            dist = np.linalg.norm(self.path[0].x[:2] - x_agent[:2])
            if dist < R_S*1.5:
                self.path.pop(0)

        new_root = self.path[0]
        self.tree.set_root(new_root)
        self.Qs.insert(0, new_root)
        root_rewire(self.tree, self.Qs, all_obstacles, self.path)

        # ---- Plan k steps (Algorithm 6) ----
        #self.path = plan_k_steps(self.tree, x_goal,x_agent)
        #self.path = greedy_path_to_goal(self.tree, x_goal)

        self.path = plan_to_goal(self.tree, x_goal, x_agent, all_obstacles,dynamic_obstacles)
        # Update best path length
        
        self.c_best = self.tree.cost(self.path[-1]) if len(self.path) > 0 else np.inf
        self.path_exists = len(self.path) > 0
        return self.path
        