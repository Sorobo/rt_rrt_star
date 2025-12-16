# tree.py

import numpy as np
import matplotlib.pyplot as plt

from node_module import Node
from spatial_index import SpatialIndex
from collision import line_collision_free
from config import K_MAX, R_S, OBSTACLE_BLOCK_RADIUS,WORLD_AREA,STEP_SIZE
from rtree_module import RTreeSpatialIndex

class Tree:
    def __init__(self, x0):
        self.root = Node(x0)
        self.index = RTreeSpatialIndex()
        self.index.insert(self.root)
        self.root.parent = None

    def nearest_node(self, x):
        return self.index.nearest(x, k=1)[0]
        

    def neighbor_radius(self):
        # ε = sqrt( (|X| * kmax)/(pi * N) )
        area = WORLD_AREA
        
        return min(max(R_S, np.sqrt(area * K_MAX / (np.pi * len(self.index.nodes)))),STEP_SIZE*5)

    def nearby(self, x):
        eps = self.neighbor_radius()

        return self.index.radius_search(x, eps)

    def add_node(self, x_new, n_closest, obstacles, rectangles=None):
        n = Node(x_new)
        n.cost = n_closest.cost + np.linalg.norm(x_new - n_closest.x)
        n_closest.add_child(n)
        n.parent = n_closest
        self.index.insert(n)
        return n

    def total_nodes(self):
        return len(self.nodes)
    
    def plot_tree(self):
        plt.close('all')
        plt.figure()  # Clear the previous plot
        for node in self.nodes:
            plt.plot(node.x[0], node.x[1], 'bo')  # Plot nodes as blue dots
        plt.draw()
        plt.pause(0.01)  # Pause to update the plot



    def set_root(self, new_root):
        # Reverse the path from new_root to current root
            path = []
            current = new_root
            while current is not None:
                path.append(current)
                current = current.parent
            
            # Reverse parent-child relationships along the path
            for i in range(len(path) - 1):
                child = path[i]
                parent = path[i + 1]
                
                # Remove child from parent's children list
                parent.children.remove(child)
                
                # Add parent as child of child
                child.children.append(parent)
                
                # Update parent pointers
                parent.parent = child
            
            # Set new root
            new_root.parent = None
            self.root = new_root
            
            # Update all costs from the new root
            self._update_costs_from_root()
        
    def _update_costs_from_root(self):
        from collections import deque
        queue = deque([self.root])
        self.root.cost = 0.0

        while queue:
            node = queue.popleft()
            for child in node.children:
                # Don't update cost for blocked nodes
                if child.cost != float("inf"):
                    child.cost = self.cost(node) + np.linalg.norm(child.x - node.x)
                    queue.append(child)
                # Blocked nodes stay blocked, don't propagate to their children


    def cost(self, node):
        if node.parent is None:
            return node.cost
        if node.cost == float("inf"):
            return float("inf")
        return node.parent.cost + np.linalg.norm(node.x - node.parent.x)
    
    def block_nodes_near_obstacle(self, obstacle_center, obstacle_radius):
        """
        Block nodes that are too close to a dynamic obstacle.
        Only checks nodes within OBSTACLE_SENSE_RADIUS for efficiency.
        Blocks a node if:
        1. The node itself is within OBSTACLE_BLOCK_RADIUS of the obstacle
        2. The edge from parent to node intersects the obstacle blocking radius
        """
        from config import OBSTACLE_BLOCK_RADIUS, OBSTACLE_SENSE_RADIUS
        
        # Get candidate nodes within sensing radius
        candidate_nodes = self.index.radius_search(obstacle_center, OBSTACLE_SENSE_RADIUS)
        count = 0
        
        for node in candidate_nodes:
            if node is self.root or node.cost == float("inf"):
                continue  # Skip root and already blocked nodes
            
            # Check if node position is within blocking radius
            node_dist = np.linalg.norm(node.x[:2] - obstacle_center)
            should_block = node_dist <= (OBSTACLE_BLOCK_RADIUS + obstacle_radius)
            
            # If node itself is not blocked, check the edge from parent
            if not should_block and node.parent is not None:
                # Check if edge from parent to node passes through blocking radius
                should_block = self._edge_intersects_circle(
                    node.parent.x[:2], node.x[:2], 
                    obstacle_center, OBSTACLE_BLOCK_RADIUS + obstacle_radius
                )
            
            if should_block:
                node.cost = float("inf")
                count += 1
                # Propagate infinite cost to all descendants
                self._propagate_infinite_cost(node)
                print("Blocked node at:", node.x," with cost:",node.cost)
        
        return count
    
    def _edge_intersects_circle(self, p1, p2, circle_center, circle_radius):
        """
        Check if line segment from p1 to p2 intersects circle.
        Uses distance from point to line segment formula.
        """
        # Vector from p1 to p2
        d = p2 - p1
        # Vector from p1 to circle center
        f = p1 - circle_center
        
        a = np.dot(d, d)
        if a < 1e-10:  # p1 == p2
            return np.linalg.norm(p1 - circle_center) <= circle_radius
        
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - circle_radius * circle_radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return False  # No intersection
        
        # Check if intersection points are on the segment [0, 1]
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        
        # Intersection occurs if either t is in [0, 1]
        return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)
    
    def _propagate_infinite_cost(self, node):
        """
        Recursively propagate infinite cost to all descendants.
        """
        children_queue = list(node.children)
        while children_queue:
            child_node = children_queue.pop(0)
            if child_node.cost != float("inf"):
                child_node.cost = float("inf")
                children_queue.extend(child_node.children)
    
    def unblock_nodes(self, all_obstacles):
        """
        Check all blocked nodes and unblock if no longer in obstacle range.
        Checks both node position and edge from parent.
        """
        from config import OBSTACLE_BLOCK_RADIUS
        
        for node_id in self.index.nodes:
            node = self.index.nodes[node_id]
            # Check nodes with infinite cost (blocked)
            if node.cost == float("inf") and node is not self.root:
                # Check if node is still blocked by any obstacle
                still_blocked = False
                
                for obs_center, obs_radius in all_obstacles:
                    # Check node position
                    node_dist = np.linalg.norm(node.x[:2] - obs_center)
                    if node_dist <= (OBSTACLE_BLOCK_RADIUS + obs_radius):
                        still_blocked = True
                        break
                    
                    # Check edge from parent
                    if node.parent is not None:
                        if self._edge_intersects_circle(
                            node.parent.x[:2], node.x[:2],
                            obs_center, OBSTACLE_BLOCK_RADIUS + obs_radius
                        ):
                            still_blocked = True
                            break
                
                if not still_blocked:
                    # Recompute cost from parent
                    if node.parent and node.parent.cost != float("inf"):
                        node.cost = node.parent.cost + np.linalg.norm(node.x - node.parent.x)
                        # Recursively update children costs
                        self._propagate_cost_update(node)
                        print("Unblocked node at:", node.x)
    
    def _propagate_cost_update(self, node):
        """
        Recursively update costs for descendants after unblocking.
        """
        children_queue = list(node.children)
        while children_queue:
            child_node = children_queue.pop(0)
            if child_node.cost == float("inf") and child_node.parent and child_node.parent.cost != float("inf"):
                # Check if this child is still blocked by obstacles
                # (it might have been blocked independently)
                child_node.cost = child_node.parent.cost + np.linalg.norm(child_node.x - child_node.parent.x)
                children_queue.extend(child_node.children)
    
    def is_node_blocked(self, all_obstacles,dyn_obstacles, node):
        for obs_center, obs_radius in all_obstacles:
            if np.linalg.norm(node.x[2:] - obs_center) <=  obs_radius:
                return True
        for dyn_obs in dyn_obstacles:
            if np.linalg.norm(node.x[2:] - dyn_obs.center) <= dyn_obs.radius+OBSTACLE_BLOCK_RADIUS:
                return True
            
        return False
    
if __name__ == "__main__":
    import numpy as np

    # Create a tree with 7 nodes spread randomly in a square of area WORLD_AREA
    if 'WORLD_AREA' in globals():
        side = int(np.sqrt(WORLD_AREA))
    else:
        side = 1  # fallback

    # Random root
    root_pos = [0.5,0.5,0.0]
    tree = Tree(root_pos)

    # Add 6 more random nodes
    for _ in range(200):
        x_new = np.array([np.random.uniform(0, 1), np.random.uniform(0, 1), 0.0])
        nearest = tree.nearest_node(x_new)
        tree.add_node(x_new, nearest, obstacles=None)
    print("Tree has", len(tree.index.nodes), "nodes.")
    # Plot the tree with arrows as edges
 

    # Plot all nodes
    for node_id in tree.index.nodes:
        node = tree.index.nodes[node_id]
        print("Node:", node.x)
        plt.plot(node.x[0], node.x[1], 'bo', markersize=8)

    # Plot edges with arrows
    for node_id in tree.index.nodes:
        node = tree.index.nodes[node_id]
        if node.parent is not None:
            plt.arrow(node.parent.x[0], node.parent.x[1],
                     node.x[0] - node.parent.x[0],
                     node.x[1] - node.parent.x[1],
                     head_width=0.01, head_length=0.03,
                     fc='red', ec='red', alpha=0.6,
                     length_includes_head=True)

    # Highlight root
    plt.plot(tree.root.x[0], tree.root.x[1], 'go', markersize=12, label='Root')

    plt.grid(True)
    plt.legend()
    plt.title('Tree Structure with Directed Edges')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    tree.set_root(tree.nearest_node(np.array([0.8,0.8,0])))
    print("New root set to:", tree.root.x)
    
    # Plot all nodes
    for node_id in tree.index.nodes:
        node = tree.index.nodes[node_id]
        print("Node:", node.x)
        plt.plot(node.x[0], node.x[1], 'bo', markersize=8)

    # Plot edges with arrows
    for node_id in tree.index.nodes:
        node = tree.index.nodes[node_id]
        if node.parent is not None:
            plt.arrow(node.parent.x[0], node.parent.x[1],
                     node.x[0] - node.parent.x[0],
                     node.x[1] - node.parent.x[1],
                     head_width=0.01, head_length=0.03,
                     fc='red', ec='red', alpha=0.6,
                     length_includes_head=True)

    # Highlight root
    plt.plot(tree.root.x[0], tree.root.x[1], 'go', markersize=12, label='Root')

    plt.grid(True)
    plt.legend()
    plt.title('Tree Structure with Directed Edges')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()