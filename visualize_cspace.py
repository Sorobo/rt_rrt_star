# visualize_cspace.py
"""
Visualize the configuration space (C-space) of the boat as a 3D point cloud.
X and Y represent position, Z represents heading angle.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collision import boat_path_collision_free
from config import WORLD_BOUNDS, BOAT_WIDTH, BOAT_LENGTH


def sample_configuration_space(obstacles, n_samples=10000):
    """
    Sample random boat configurations and check collision.
    
    Parameters
    ----------
    obstacles : list
        List of (center, radius) tuples
    n_samples : int
        Number of random configurations to sample
        
    Returns
    -------
    free_configs : np.array
        Array of collision-free configurations (n x 3) [x, y, heading]
    collision_configs : np.array
        Array of configurations in collision (n x 3) [x, y, heading]
    """
    x_min, x_max = WORLD_BOUNDS[0]
    y_min, y_max = WORLD_BOUNDS[1]
    
    free_configs = []
    collision_configs = []
    
    print(f"Sampling {n_samples} configurations...")
    
    for i in range(n_samples):
        if i % 1000 == 0:
            print(f"  Progress: {i}/{n_samples}")
        
        # Sample random configuration
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        heading = np.random.uniform(-np.pi, np.pi)
        
        pos = np.array([x, y])
        
        # Check if boat at this configuration collides with any obstacle
        # We check the boat at this single position (start = end)
        is_free = boat_path_collision_free(
            pos, heading, pos, heading,
            BOAT_WIDTH, BOAT_LENGTH, obstacles
        )
        
        config = [x, y, heading]
        
        if is_free:
            free_configs.append(config)
        else:
            collision_configs.append(config)
    
    print(f"Done! Free: {len(free_configs)}, Collision: {len(collision_configs)}")
    
    return np.array(free_configs), np.array(collision_configs)


def visualize_cspace_3d(free_configs, collision_configs, obstacles):
    """
    Visualize the configuration space as a 3D point cloud.
    
    Parameters
    ----------
    free_configs : np.array
        Collision-free configurations (n x 3)
    collision_configs : np.array
        Configurations in collision (n x 3)
    obstacles : list
        List of obstacles for reference
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot free space (green)
    if len(free_configs) > 0:
        ax.scatter(free_configs[:, 0], free_configs[:, 1], free_configs[:, 2],
                  c='green', marker='.', s=3, alpha=1, label='Free')
    
    # Plot collision space (red)
    """
    if len(collision_configs) > 0:
        ax.scatter(collision_configs[:, 0], collision_configs[:, 1], collision_configs[:, 2],
                  c='red', marker='.', s=1, alpha=0.3, label='Collision')
    """
    # Labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Heading (radians)')
    ax.set_title(f'Configuration Space Visualization\n'
                f'Free: {len(free_configs)}, Collision: {len(collision_configs)}')
    
    # Set z-axis limits to heading range
    ax.set_zlim(-np.pi, np.pi)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create second figure: 2D slice at heading = 0
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Filter configurations near heading = 0
    heading_tolerance = 0.2  # radians (~11 degrees)
    free_slice = free_configs[np.abs(free_configs[:, 2]) < heading_tolerance]
    collision_slice = collision_configs[np.abs(collision_configs[:, 2]) < heading_tolerance]
    
    if len(free_slice) > 0:
        ax2.scatter(free_slice[:, 0], free_slice[:, 1], 
                   c='green', marker='.', s=5, alpha=0.5, label='Free')
    
    if len(collision_slice) > 0:
        ax2.scatter(collision_slice[:, 0], collision_slice[:, 1],
                   c='red', marker='.', s=5, alpha=0.5, label='Collision')
    
    # Draw obstacles
    for center, radius in obstacles:
        circle = plt.Circle(center, radius, fill=False, edgecolor='blue', 
                          linewidth=2, linestyle='--')
        ax2.add_patch(circle)
    
    ax2.set_xlim(WORLD_BOUNDS[0])
    ax2.set_ylim(WORLD_BOUNDS[1])
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title(f'2D Slice at Heading ≈ 0° (±{np.rad2deg(heading_tolerance):.1f}°)')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to generate and visualize C-space."""
    
    # Define obstacles (same as in demo)
    obstacles = [
        (np.array([20.0, 20.0]), 5.0),
        (np.array([50.0, 50.0]), 8.0),
        (np.array([70.0, 30.0]), 6.0),
        (np.array([30.0, 70.0]), 5.0),
        (np.array([80.0, 80.0]), 7.0),
    ]
    
    # Sample configuration space
    n_samples = 5000  # Adjust this number based on your computer's speed
    free_configs, collision_configs = sample_configuration_space(obstacles, n_samples)
    
    # Visualize
    visualize_cspace_3d(free_configs, collision_configs, obstacles)


if __name__ == "__main__":
    main()
