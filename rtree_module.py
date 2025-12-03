import rtree.index as index
import numpy as np


class RTreeSpatialIndex:
    """
    Drop-in replacement for the grid-based spatial index.
    Stores node bounding boxes in an R-tree.
    """

    def __init__(self, position_weight=1.0, heading_weight=1, velocity_weight=5):
        # Rtree configuration
        p = index.Property()
        p.dimension = 6  # 6D space (x, y, theta, x_dot, y_dot, theta_dot)
        self.idx = index.Index(properties=p)

        # Keep mapping from Rtree IDs -> node objects
        self.nodes = {}
        self.next_id = 0
        
        # Scaling weights for different dimensions
        self.position_weight = position_weight  # Weight for x, y
        self.heading_weight = heading_weight    # Weight for theta
        self.velocity_weight = velocity_weight  # Weight for x_dot, y_dot, theta_dot
    
    def _scale_coords(self, point):
        """
        Scale coordinates to weight different parameters.
        
        Parameters
        ----------
        point : np.array
            Point [x, y, theta, x_dot, y_dot, theta_dot] (6D state)
        
        Returns
        -------
        tuple
            Scaled (x, y, theta, x_dot, y_dot, theta_dot)
        """
        x = float(point[0]) * self.position_weight
        y = float(point[1]) * self.position_weight
        theta = float(point[2]) * self.heading_weight if len(point) > 2 else 0.0
        
        # Include velocities if available (6D state)
        if len(point) > 3:
            x_dot = float(point[3]) * self.velocity_weight
            y_dot = float(point[4]) * self.velocity_weight
            theta_dot = float(point[5]) * self.velocity_weight if len(point) > 5 else 0.0
        else:
            x_dot = 0.0
            y_dot = 0.0
            theta_dot = 0.0
        
        return x, y, theta, x_dot, y_dot, theta_dot

    def insert(self, node):
        """
        Insert a node into the R-tree.
        Node must have an attribute node.x as a state vector.
        """
        nid = self.next_id
        self.next_id += 1

        x, y, theta, x_dot, y_dot, theta_dot = self._scale_coords(node.x)

        # Rtree requires bounding boxes (min coords, max coords)
        bbox = (x, y, theta, x_dot, y_dot, theta_dot, 
                x, y, theta, x_dot, y_dot, theta_dot)

        self.idx.insert(nid, bbox)
        self.nodes[nid] = node
        node._rtree_id = nid  # store for deletion

    def remove(self, node):
        """Remove a node from the spatial index."""
        nid = node._rtree_id
        x, y, theta, x_dot, y_dot, theta_dot = self._scale_coords(node.x)
        bbox = (x, y, theta, x_dot, y_dot, theta_dot,
                x, y, theta, x_dot, y_dot, theta_dot)
        self.idx.delete(nid, bbox)
        del self.nodes[nid]

    def nearest(self, point, k=1):
        """
        Return k nearest nodes to a point.
        point is np.array([x, y, theta, x_dot, y_dot, theta_dot])
        """
        x, y, theta, x_dot, y_dot, theta_dot = self._scale_coords(point)
        bbox = (x, y, theta, x_dot, y_dot, theta_dot,
                x, y, theta, x_dot, y_dot, theta_dot)
        results = list(self.idx.nearest(bbox, k))

        return [self.nodes[i] for i in results]

    def radius_search(self, point, radius):
        """
        Return all nodes within a given radius.
        Note: radius is applied in scaled space.
        """
        x, y, theta, x_dot, y_dot, theta_dot = self._scale_coords(point)
        bbox = (x - radius, y - radius, theta - radius, 
                x_dot - radius, y_dot - radius, theta_dot - radius,
                x + radius, y + radius, theta + radius,
                x_dot + radius, y_dot + radius, theta_dot + radius)

        # this gives bounding box candidates; we check distance manually
        candidates = self.idx.intersection(bbox)

        out = []
        for nid in candidates:
            node = self.nodes[nid]
            # Compute distance in scaled space
            node_scaled = np.array(self._scale_coords(node.x))
            point_scaled = np.array([x, y, theta, x_dot, y_dot, theta_dot])
            if np.linalg.norm(node_scaled - point_scaled) <= radius:
                out.append(node)

        return out
