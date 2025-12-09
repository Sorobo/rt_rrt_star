import rtree.index as index
import numpy as np


class RTreeSpatialIndex:
    """
    Drop-in replacement for the grid-based spatial index.
    Stores node bounding boxes in an R-tree.
    """

    def __init__(self):
        # Rtree configuration
        p = index.Property()
        p.dimension = 3  # 3D space
        self.idx = index.Index(properties=p)

        # Keep mapping from Rtree IDs -> node objects
        self.nodes = {}
        self.next_id = 0

    def insert(self, node):
        """
        Insert a node into the R-tree.
        Node must have an attribute node.x as a 2D numpy array.
        """
        nid = self.next_id
        self.next_id += 1

        x, y,theta = float(node.x[0]), float(node.x[1]), float(node.x[2])

        # Rtree requires bounding boxes (minx, miny, maxx, maxy)
        bbox = (x, y, theta*5, x, y, theta*5)

        self.idx.insert(nid, bbox)
        self.nodes[nid] = node
        node._rtree_id = nid  # store for deletion

    def remove(self, node):
        """Remove a node from the spatial index."""
        nid = node._rtree_id
        x, y, theta = float(node.x[0]), float(node.x[1]), float(node.x[2])
        bbox = (x, y, theta*5, x, y, theta*5)
        self.idx.delete(nid, bbox)
        del self.nodes[nid]

    def nearest(self, point, k=1):
        """
        Return k nearest nodes to a point.
        point is np.array([x,y])
        """
        x, y, theta = float(point[0]), float(point[1]), float(point[2])
        results = list(self.idx.nearest((x, y, theta*5, x, y, theta*5), k))

        return [self.nodes[i] for i in results]

    def radius_search(self, point, radius):
        """
        Return all nodes within a given radius.
        """
        x, y, theta = float(point[0]), float(point[1]), float(point[2])
        bbox = (x - radius, y - radius, theta*5 - radius, x + radius, y + radius, theta*5 + radius)

        # this gives bounding box candidates; we check distance manually
        candidates = self.idx.intersection(bbox)

        out = []
        for nid in candidates:
            node = self.nodes[nid]
            if np.linalg.norm(node.x - point) <= radius:
                out.append(node)

        return out
