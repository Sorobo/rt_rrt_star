# tree.py

import numpy as np
import matplotlib.pyplot as plt

from node_module import Node
from spatial_index import SpatialIndex
from collision import line_collision_free
from config import K_MAX, R_S, OBSTACLE_BLOCK_RADIUS,WORLD_AREA,STEP_SIZE
from rtree_module import RTreeSpatialIndex

class Tree:
    def __init__(self, x0, rtree_weights=None):
        self.root = Node(x0)
        if rtree_weights:
            self.index = RTreeSpatialIndex(**rtree_weights)
        else:
            self.index = RTreeSpatialIndex()
        self.index.insert(self.root)
        self.root.parent = None

    def nearest_node(self, x, skip_ineligible=True):
        """Find nearest node, optionally skipping ineligible nodes"""
        if skip_ineligible:
            # Search for multiple candidates and return first eligible one
            candidates = self.index.nearest(x, k=10)
            for candidate in candidates:
                if not candidate.ineligible:
                    return candidate
            # If all are ineligible, return the nearest anyway (fallback)
            return candidates[0] if candidates else self.root
        return self.index.nearest(x, k=1)[0]
        

    def neighbor_radius(self):
        # ε = sqrt( (|X| * kmax)/(pi * N) )
        area = WORLD_AREA
        
        return min(max(R_S, np.sqrt(area * K_MAX / (np.pi * len(self.index.nodes)))),STEP_SIZE*5)

    def nearby(self, x, skip_ineligible=True):
        """Find nearby nodes, optionally filtering out ineligible ones"""
        eps = self.neighbor_radius()
        nodes = self.index.radius_search(x, eps)
        
        if skip_ineligible:
            return [n for n in nodes if not n.ineligible]
        return nodes

    def add_node(self, x_new, n_closest, obstacles, control=None):
        """
        Add a new node to the tree.
        
        Parameters
        ----------
        x_new : np.array
            New state (can be 2D, 3D, or 6D)
        n_closest : Node
            Parent node
        obstacles : list
            List of obstacles
        control : np.array, optional
            Control input that led to this state
        
        Returns
        -------
        Node
            The newly created node
        """
        from rewiring import dynamic_cost_heuristic
        n = Node(x_new)
        
        # Use dynamic cost heuristic instead of simple distance
        n.cost = n_closest.cost + dynamic_cost_heuristic(n_closest, n)
        
        # Store control if provided
        if control is not None:
            n.control = control
        
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
        from rewiring import dynamic_cost_heuristic
        queue = deque([self.root])
        self.root.cost = 0.0

        while queue:
            node = queue.popleft()
            for child in node.children:
                child.cost = self.cost(node) + dynamic_cost_heuristic(node, child)
                queue.append(child)


    def cost(self, node):
        from rewiring import dynamic_cost_heuristic
        if node.parent is None:
            return node.cost
        if node.blocked:
            return float("inf")
        return node.parent.cost + dynamic_cost_heuristic(node.parent, node)
    
    def block_nodes_near_obstacle(self, obstacle_center, block_radius):
        """
        Set cost to inf for all nodes within block_radius of obstacle_center.
        Returns the number of nodes blocked.
        """
        affected_nodes = self.index.radius_search(obstacle_center, block_radius)
        count = 0
        
        for node in affected_nodes:
            if node is not self.root:  # Don't block the root
                node.cost = float("inf")
                node.blocked = True
                count += 1
                children_queue = [child for child in node.children]
                while children_queue:
                    child_node = children_queue.pop(0)
                    if not child_node.blocked:
                        child_node.cost = float("inf")
                        count += 1
                        children_queue.extend(child_node.children)
        
        return count
    
    def unblock_nodes(self, all_obstacles):
        """
        Check all blocked nodes and unblock if no longer in obstacle range.
        Recompute their costs if unblocked.
        """
        for node_id in self.index.nodes:
            node = self.index.nodes[node_id]
            if node.blocked:
                # Check if node is still blocked by any obstacle
                still_blocked = False
                for obs_center, obs_radius in all_obstacles:
                    if np.linalg.norm(node.x - obs_center) <= OBSTACLE_BLOCK_RADIUS + obs_radius:
                        still_blocked = True
                        break
                
                if not still_blocked:
                    from rewiring import dynamic_cost_heuristic
                    node.blocked = False
                    # Recompute cost from parent
                    if node.parent and node.parent.cost != float("inf"):
                        node.cost = node.parent.cost + dynamic_cost_heuristic(node.parent, node)
                    else:
                        node.cost = float("inf")
    
    def is_node_blocked(self, all_obstacles,dyn_obstacles, node):
        for obs_center, obs_radius in all_obstacles:
            if np.linalg.norm(node.x[:2] - obs_center) <=  obs_radius:
                return True
        for dyn_obs in dyn_obstacles:
            if np.linalg.norm(node.x[:2] - dyn_obs.center) <= dyn_obs.radius+OBSTACLE_BLOCK_RADIUS:
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