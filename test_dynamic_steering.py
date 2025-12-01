# test_dynamic_steering.py
"""
Test script for dynamic steering with boat dynamics.
Demonstrates the steer_dynamic function.
"""

import numpy as np
import matplotlib.pyplot as plt
from steer import steer_dynamic
from config import BOAT_WIDTH, BOAT_LENGTH

def test_dynamic_steering():
    """Test dynamic steering with visualization."""
    
    # Define start and target states
    # State = [x, y, theta, x_dot, y_dot, theta_dot]
    start_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target_state = np.array([10.0, 5.0, np.pi/4, 0.0, 0.0, 0.0])
    
    print("Testing Dynamic Steering")
    print("=" * 50)
    print(f"Start state: {start_state}")
    print(f"Target state: {target_state}")
    print("\nComputing optimal control...")
    
    # Compute optimal trajectory
    final_state, optimal_control = steer_dynamic(start_state, target_state, dt=0.1, horizon=20)
    
    print(f"\nOptimal control: {optimal_control}")
    print(f"Final state: {final_state}")
    print(f"Position error: {np.linalg.norm(final_state[:2] - target_state[:2]):.3f}")
    print(f"Heading error: {abs(final_state[2] - target_state[2]):.3f} rad")
    
    # Simulate full trajectory for visualization
    trajectory = simulate_trajectory(start_state, optimal_control, dt=0.1, steps=20)
    
    # Visualize
    visualize_steering(trajectory, start_state, target_state, optimal_control)


def simulate_trajectory(start_state, control, dt, steps):
    """Simulate trajectory with constant control."""
    # Boat parameters
    m = 1800.0
    Iz = 4860.0
    Xu = -157.0
    Yv = -258.0
    Nr = -1000.0
    
    trajectory = [start_state.copy()]
    state = start_state.copy()
    
    for _ in range(steps):
        x, y, theta, u, v, r = state
        tau_x, tau_y, tau_n = control
        
        # Rotation matrix
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        
        # Accelerations
        u_dot = (tau_x + Xu * u) / m
        v_dot = (tau_y + Yv * v) / m
        r_dot = (tau_n + Nr * r) / Iz
        
        # Position derivatives
        x_dot = cos_th * u - sin_th * v
        y_dot = sin_th * u + cos_th * v
        theta_dot = r
        
        # Euler integration
        state += np.array([x_dot, y_dot, theta_dot, u_dot, v_dot, r_dot]) * dt
        state[2] = np.arctan2(np.sin(state[2]), np.cos(state[2]))  # Normalize angle
        
        trajectory.append(state.copy())
    
    return np.array(trajectory)


def visualize_steering(trajectory, start_state, target_state, control):
    """Visualize the steering result."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: XY trajectory
    ax = axes[0, 0]
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
    ax.plot(start_state[0], start_state[1], 'go', markersize=10, label='Start')
    ax.plot(target_state[0], target_state[1], 'ro', markersize=10, label='Target')
    
    # Draw boat at start and end
    for i, (state, color) in enumerate([(start_state, 'green'), (trajectory[-1], 'blue')]):
        corners = get_boat_corners(state[:3])
        corners = np.vstack([corners, corners[0]])  # Close the polygon
        ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=2)
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('XY Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Heading over time
    ax = axes[0, 1]
    time = np.arange(len(trajectory)) * 0.1
    ax.plot(time, np.rad2deg(trajectory[:, 2]), 'b-', linewidth=2)
    ax.axhline(np.rad2deg(target_state[2]), color='r', linestyle='--', label='Target')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Heading [deg]')
    ax.set_title('Heading vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Velocities over time
    ax = axes[1, 0]
    ax.plot(time, trajectory[:, 3], label='u (surge)', linewidth=2)
    ax.plot(time, trajectory[:, 4], label='v (sway)', linewidth=2)
    ax.plot(time, trajectory[:, 5], label='r (yaw rate)', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity')
    ax.set_title('Velocities vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Control inputs
    ax = axes[1, 1]
    ax.bar(['τ_x', 'τ_y', 'τ_n'], control, color=['red', 'green', 'blue'])
    ax.set_ylabel('Control Input')
    ax.set_title('Optimal Control Inputs')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def get_boat_corners(state):
    """Get boat corners for visualization."""
    x, y, theta = state[:3]
    
    half_length = BOAT_LENGTH / 2
    half_width = BOAT_WIDTH / 2
    
    local_corners = np.array([
        [half_length, half_width],
        [half_length, -half_width],
        [-half_length, -half_width],
        [-half_length, half_width]
    ])
    
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    rotation = np.array([[cos_th, -sin_th],
                        [sin_th, cos_th]])
    
    return local_corners @ rotation.T + np.array([x, y])


if __name__ == "__main__":
    test_dynamic_steering()
