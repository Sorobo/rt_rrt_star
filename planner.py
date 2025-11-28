# planner.py

import numpy as np
from config import K_PLANNING, GOAL_RADIUS
from node_module import Node

def plan_k_steps(tree, x_goal,x_agent):
    print("planning k steps")
    closest_to_goal = tree.nearest_node(x_goal)
    if not closest_to_goal.blocked:
        if np.linalg.norm(closest_to_goal.x - x_goal) < GOAL_RADIUS:
            path = []
            current = closest_to_goal
            while current is not None:
                # Skip blocked nodes
                if current.cost == float("inf"):
                    print("returning path of length", 0)
                    return greedy_path_to_goal(tree, x_goal)  # Path is blocked
                path.append(current)
                current = current.parent
                if current is not None:
                    if current.parent is not None:
                        if current.parent == current:
                            current.parent = None
        
            path.reverse()
            print("oioioioi")
            print("returning path of length", len(path))
            return path
    
    # Greedy path from root toward the node closest to the goal
    return greedy_path_to_goal(tree, x_goal)

def greedy_path_to_goal(tree, x_goal):
    print("started greedy to goal")
    current = tree.root
    path = [current]
    while len(path) < K_PLANNING:
        # Select child closest to goal that is not blocked
        unblocked_children = [child for child in current.children if not child.blocked]
        if not unblocked_children:
            break  # No unblocked children, stop path extension
        next_node = min(unblocked_children, key=lambda n: np.linalg.norm(n.x - x_goal))
        path.append(next_node)
        current = next_node
        if np.linalg.norm(current.x - x_goal) < GOAL_RADIUS:
            break  # Reached goal vicinity
    return path

def backtrack_down_tree(tree, start_node):
    path = []
    current = start_node
    while current is not None and current not in path:
        path.append(current)
        current = current.parent
    path.reverse()
    return path


def plan_to_goal(tree, x_goal, x_agent,all_obstacles,dyn_obstacles):
    #if tree.is_node_blocked(all_obstacles,dyn_obstacles,Node(x_agent)):
    #    return [Node(x_agent)]
    
    if tree.is_node_blocked(all_obstacles,dyn_obstacles,Node(x_goal)):
        return greedy_path_to_goal(tree, x_goal)
        
    closest_to_goal = tree.nearest_node(x_goal)
    
    if tree.cost(closest_to_goal) == float("inf"):
        return greedy_path_to_goal(tree, x_goal)
    
    if np.linalg.norm(closest_to_goal.x - x_goal) < GOAL_RADIUS:
        return backtrack_down_tree(tree, closest_to_goal)
        
    else:
        #use greedy path to goal
        new_goal = tree.nearest_node(x_goal)
        return backtrack_down_tree(tree, new_goal)
        return greedy_path_to_goal(tree, new_goal.x)
        