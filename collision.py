# collision.py

import numpy as np
from scipy.spatial import ConvexHull
from config import BOAT_WIDTH, BOAT_LENGTH,WORLD_BOUNDS


def line_collision_free(x1, x2, obstacles):
    """Very simple checker: return True if straight line is collision-free."""
    for obs in obstacles:
        center, radius = obs
        # Distance from segment to obstacle center
        v = x2 - x1
        w = center - x1
        t = max(0, min(1, np.dot(v, w) / np.dot(v, v)))
        closest = x1 + t * v
        if np.linalg.norm(closest - center) < radius:
            return False
    return True


def get_boat_corners(position, heading, width, length):
    """
    Get the four corners of a rectangular boat.
    
    Parameters
    ----------
    position : np.array
        Center position [x, y]
    heading : float
        Heading angle in radians
    width : float
        Boat width
    length : float
        Boat length
        
    Returns
    -------
    np.array
        Array of shape (4, 2) containing corner positions
    """
    half_length = length / 2.0
    half_width = width / 2.0
    
    # Local coordinates of corners
    local_corners = np.array([
        [half_length, half_width],    # Front right
        [half_length, -half_width],   # Front left
        [-half_length, -half_width],  # Back left
        [-half_length, half_width]    # Back right
    ])
    
    # Rotation matrix
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    rotation = np.array([[cos_h, -sin_h],
                        [sin_h, cos_h]])
    
    # Transform to global coordinates
    global_corners = local_corners @ rotation.T + position
    
    return global_corners


def create_swept_hull(start_pos, start_heading, end_pos, end_heading, width, length):
    """
    Create a convex hull around the boat's start and end positions.
    
    Parameters
    ----------
    start_pos : np.array
        Starting position [x, y]
    start_heading : float
        Starting heading in radians
    end_pos : np.array
        Ending position [x, y]
    end_heading : float
        Ending heading in radians
    width : float
        Boat width
    length : float
        Boat length
        
    Returns
    -------
    np.array
        Vertices of the convex hull in counter-clockwise order
    """
    # Get corners at start position
    start_corners = get_boat_corners(start_pos, start_heading, width, length)
    
    # Get corners at end position
    end_corners = get_boat_corners(end_pos, end_heading, width, length)
    
    # Combine all 8 points
    all_points = np.vstack([start_corners, end_corners])
    
    # Compute convex hull
    hull = ConvexHull(all_points)
    
    # Return vertices in order
    return all_points[hull.vertices]


def sat_polygon_circle(polygon_vertices, circle_center, circle_radius):
    """
    Check collision between a polygon and a circle using Separating Axis Theorem.
    
    Parameters
    ----------
    polygon_vertices : np.array
        Vertices of the polygon (N x 2)
    circle_center : np.array
        Center of the circle [x, y]
    circle_radius : float
        Radius of the circle
        
    Returns
    -------
    bool
        True if collision detected, False otherwise
    """
    n_vertices = len(polygon_vertices)
    
    # Test all polygon edge normals
    for i in range(n_vertices):
        # Get edge
        p1 = polygon_vertices[i]
        p2 = polygon_vertices[(i + 1) % n_vertices]
        edge = p2 - p1
        
        # Get perpendicular (normal)
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)
        
        # Project polygon vertices onto axis
        poly_projections = polygon_vertices @ normal
        poly_min = np.min(poly_projections)
        poly_max = np.max(poly_projections)
        
        # Project circle onto axis
        circle_projection = circle_center @ normal
        circle_min = circle_projection - circle_radius
        circle_max = circle_projection + circle_radius
        
        # Check for separation
        if poly_max < circle_min or circle_max < poly_min:
            return False  # Separating axis found, no collision
    
    # Test axis from polygon vertices to circle center
    for vertex in polygon_vertices:
        axis = circle_center - vertex
        axis_length = np.linalg.norm(axis)
        
        if axis_length < 1e-10:
            continue  # Skip if vertex is at circle center
            
        axis = axis / axis_length
        
        # Project polygon onto axis
        poly_projections = polygon_vertices @ axis
        poly_min = np.min(poly_projections)
        poly_max = np.max(poly_projections)
        
        # Project circle onto axis
        circle_projection = circle_center @ axis
        circle_min = circle_projection - circle_radius
        circle_max = circle_projection + circle_radius
        
        # Check for separation
        if poly_max < circle_min or circle_max < poly_min:
            return False  # Separating axis found, no collision
    
    return True  # No separating axis found, collision detected


def boat_path_collision_free(start_pos, start_heading, end_pos, end_heading, 
                             width, length, obstacles):
    """
    Check if a boat path from start to end is collision-free using SAT.
    
    Parameters
    ----------
    start_pos : np.array
        Starting position [x, y]
    start_heading : float
        Starting heading in radians
    end_pos : np.array
        Ending position [x, y]
    end_heading : float
        Ending heading in radians
    width : float
        Boat width
    length : float
        Boat length
    obstacles : list
        List of (center, radius) tuples
        
    Returns
    -------
    bool
        True if path is collision-free, False otherwise
    """
    # Create swept hull
    hull_vertices = create_swept_hull(start_pos, start_heading, 
                                      end_pos, end_heading, 
                                      width, length)
    
    # Check collision with each obstacle
    for obstacle in obstacles:
        center, radius = obstacle
        if sat_polygon_circle(hull_vertices, center, radius):
            return False  # Collision detected
    # Check if the hull is within world bounds
    if np.any(hull_vertices[:, 0] < WORLD_BOUNDS[0][0]) or np.any(hull_vertices[:, 0] > WORLD_BOUNDS[0][1]) or \
       np.any(hull_vertices[:, 1] < WORLD_BOUNDS[1][0]) or np.any(hull_vertices[:, 1] > WORLD_BOUNDS[1][1]):
        return False  # Hull is out of bounds
    
    
    
    return True  # No collisions


    
    
def boat_collision_free(start_node, end_node, obstacles, 
                        width=BOAT_WIDTH, length=BOAT_LENGTH):
    """
    Check if boat movement from start_node to end_node is collision-free.
    Uses boat dimensions from config if not specified.
    
    Parameters
    ----------
    start_node : Node or np.array
        Starting node (with .x and .heading) or position array
    end_node : Node or np.array  
        Ending node (with .x and .heading) or position array
    obstacles : list
        List of (center, radius) tuples
    width : float, optional
        Boat width (default from config)
    length : float, optional
        Boat length (default from config)
        
    Returns
    -------
    bool
        True if path is collision-free, False otherwise
    """
    # Extract position and heading from nodes or arrays
    start_pos = start_node.x[:2]
    start_heading = start_node.x[2]
    end_pos = end_node.x[:2]
    end_heading = end_node.x[2]
    
    return boat_path_collision_free(start_pos, start_heading, 
                                    end_pos, end_heading,
                                    width, length, obstacles)

