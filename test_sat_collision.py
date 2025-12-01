# test_sat_collision.py
"""
Test script for Separating Axis Theorem collision detection.
Demonstrates boat swept hull collision checking.
"""

import numpy as np
import matplotlib.pyplot as plt
from collision import (create_swept_hull, sat_polygon_circle, 
                      boat_path_collision_free, get_boat_corners)
from config import BOAT_WIDTH, BOAT_LENGTH

def visualize_collision_check():
    """Visualize boat swept hull and obstacle collision detection."""
    
    # Define boat path
    start_pos = np.array([5.0, 5.0])
    start_heading = np.pi/2  # Pointing up
    end_pos = np.array([15.0, 5.0])
    end_heading = 0  # Pointing up
    
    # Define obstacles
    obstacles = [
        (np.array([10.0, 7.0]), 2.0),   # Collision
        (np.array([8.0, 12.0]), 1.5),   # No collision
        (np.array([17.0, 8.0]), 1.0),   # No collision
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw boat at start position
    start_corners = get_boat_corners(start_pos, start_heading, BOAT_WIDTH, BOAT_LENGTH)
    start_polygon = plt.Polygon(start_corners, fill=False, edgecolor='blue', 
                               linewidth=2, linestyle='--', label='Start')
    ax.add_patch(start_polygon)
    
    # Draw boat at end position
    end_corners = get_boat_corners(end_pos, end_heading, BOAT_WIDTH, BOAT_LENGTH)
    end_polygon = plt.Polygon(end_corners, fill=False, edgecolor='green', 
                             linewidth=2, linestyle='--', label='End')
    ax.add_patch(end_polygon)
    
    # Draw swept hull
    hull_vertices = create_swept_hull(start_pos, start_heading, 
                                     end_pos, end_heading,
                                     BOAT_WIDTH, BOAT_LENGTH)
    hull_polygon = plt.Polygon(hull_vertices, fill=True, facecolor='cyan', 
                              alpha=0.3, edgecolor='darkblue', 
                              linewidth=2, label='Swept Hull')
    ax.add_patch(hull_polygon)
    
    # Draw obstacles and check collisions
    for i, (center, radius) in enumerate(obstacles):
        collision = sat_polygon_circle(hull_vertices, center, radius)
        color = 'red' if collision else 'gray'
        label = f'Obstacle {i+1} ({"Collision" if collision else "Clear"})'
        
        circle = plt.Circle(center, radius, fill=True, facecolor=color, 
                          alpha=0.3, edgecolor=color, linewidth=2, label=label)
        ax.add_patch(circle)
        
        # Draw center point
        ax.plot(center[0], center[1], 'k+', markersize=10)
    
    # Draw heading arrows
    arrow_length = BOAT_LENGTH * 0.6
    ax.arrow(start_pos[0], start_pos[1], 
            arrow_length * np.cos(start_heading), 
            arrow_length * np.sin(start_heading),
            head_width=0.5, head_length=0.7, fc='blue', ec='blue')
    
    ax.arrow(end_pos[0], end_pos[1], 
            arrow_length * np.cos(end_heading), 
            arrow_length * np.sin(end_heading),
            head_width=0.5, head_length=0.7, fc='green', ec='green')
    
    # Check overall collision
    is_collision_free = boat_path_collision_free(start_pos, start_heading,
                                                 end_pos, end_heading,
                                                 BOAT_WIDTH, BOAT_LENGTH,
                                                 obstacles)
    
    # Set plot properties
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 15)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    
    title = f'Boat Swept Hull Collision Detection\n'
    title += f'Path is {"COLLISION FREE" if is_collision_free else "BLOCKED"}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def test_multiple_scenarios():
    """Test multiple collision scenarios."""
    
    print("Testing SAT Collision Detection")
    print("=" * 50)
    
    # Test 1: Clear path
    result1 = boat_path_collision_free(
        np.array([0, 0]), 0.0,
        np.array([10, 0]), 0.0,
        BOAT_WIDTH, BOAT_LENGTH,
        [(np.array([5, 10]), 2.0)]
    )
    print(f"Test 1 (Clear path): {'PASS' if result1 else 'FAIL'}")
    
    # Test 2: Direct collision
    result2 = boat_path_collision_free(
        np.array([0, 0]), 0.0,
        np.array([10, 0]), 0.0,
        BOAT_WIDTH, BOAT_LENGTH,
        [(np.array([5, 0]), 2.0)]
    )
    print(f"Test 2 (Direct collision): {'PASS' if not result2 else 'FAIL'}")
    
    # Test 3: Swept hull collision (diagonal path)
    result3 = boat_path_collision_free(
        np.array([0, 0]), 0.0,
        np.array([10, 10]), np.pi/4,
        BOAT_WIDTH, BOAT_LENGTH,
        [(np.array([5, 5]), 1.5)]
    )
    print(f"Test 3 (Swept hull collision): {'PASS' if not result3 else 'FAIL'}")
    
    # Test 4: Rotation sweep collision
    result4 = boat_path_collision_free(
        np.array([5, 5]), 0.0,
        np.array([5, 5]), np.pi,
        BOAT_WIDTH, BOAT_LENGTH,
        [(np.array([8, 5]), 1.0)]
    )
    print(f"Test 4 (Rotation sweep): {'PASS' if not result4 else 'FAIL'}")
    
    print("=" * 50)


if __name__ == "__main__":
    # Run tests
    test_multiple_scenarios()
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_collision_check()
