import numpy as np
from scipy.optimize import minimize
from config import STEP_SIZE

def steer(x_rand, n_closest):
    """
    Simple geometric steering (kinematic).
    Steers toward x_rand with maximum step size.
    """
    direction = x_rand[:2] - n_closest.x[:2]
    distance = np.linalg.norm(direction)
    if distance <= STEP_SIZE:
        return x_rand
    else:
        return n_closest.x[:2] + (direction / distance) * STEP_SIZE


# Global cost function weights (optimized via automated tuning - 2025-12-03)
COST_WEIGHTS = {
    'position_weight': 15.116731097089193,
    'heading_weight': 3.443507809982464,
    'velocity_weight': 0.5126958016989883,
    'progress_weight': 4.749025576411771
}

def set_cost_weights(**kwargs):
    """Update cost function weights for parameter tuning"""
    global COST_WEIGHTS
    COST_WEIGHTS.update(kwargs)

def get_cost_weights():
    """Get current cost function weights"""
    return COST_WEIGHTS.copy()

def steer_dynamic(start_state, target_state, obstacles, dt=0.1, horizon=10):
    """
    Dynamic steering using MilliAmpere1Sim boat dynamics.
    Tries predefined maximum control inputs and picks the best one.
    
    Parameters
    ----------
    start_state : np.array
        Starting state [x, y, theta, x_dot, y_dot, theta_dot]
    target_state : np.array
        Target state (at least [x, y, theta])
    obstacles : list
        List of obstacles for collision checking
    dt : float
        Time step for integration
    horizon : int
        Number of time steps to simulate
    
    Returns
    -------
    new_state : np.array
        Resulting state after applying best control
    control : np.array
        Best control input [tau_x, tau_y, tau_n] (forces/torque)
    """
    from boat_dynamics import MilliAmpere1Sim
    from collision import boat_collision_free
    from node_module import Node
    
    # Ensure we have 6D states
    if len(start_state) < 6:
        start_state = np.pad(start_state, (0, 6 - len(start_state)), 'constant')
    if len(target_state) < 6:
        target_state = np.pad(target_state, (0, 6 - len(target_state)), 'constant')
    
    # Create a boat simulator instance (we only need it for the methods)
    boat = MilliAmpere1Sim(start_pos=start_state[:2], dt=dt)
    
    # Get predefined control options
    control_options = boat.get_max_control_options()
    
    def cost_function(final_state):
        """Compute cost of reaching a final state"""
        # Position error - MOST IMPORTANT: how close did we get?
        pos_error = np.linalg.norm(final_state[:2] - target_state[:2])
        
        # Heading error - moderate importance
        angle_diff = np.arctan2(np.sin(final_state[2] - target_state[2]), 
                               np.cos(final_state[2] - target_state[2]))
        heading_error = abs(angle_diff)
        
        # Velocity error - LOW importance (we mostly care about zero velocity at goal)
        # Only penalize if target velocity is zero (stationary goal)
        vel_error = np.linalg.norm(final_state[3:6] - target_state[3:6])
        
        # Progress toward goal - reward getting closer
        start_distance = np.linalg.norm(start_state[:2] - target_state[:2])
        final_distance = np.linalg.norm(final_state[:2] - target_state[:2])
        progress = start_distance - final_distance  # positive if we got closer
        
        # Weighted cost (minimize = better)
        # Use global weights (can be tuned)
        total_cost = (
            COST_WEIGHTS['position_weight'] * pos_error +
            COST_WEIGHTS['heading_weight'] * heading_error +
            COST_WEIGHTS['velocity_weight'] * vel_error -
            COST_WEIGHTS['progress_weight'] * progress
        )
        
        return total_cost
    
    # Try all control options and pick the best
    # Collect all (cost, control, final_state) tuples
    candidates = []
    for control in control_options:
        final_state = boat.simulate_with_control(start_state, control, dt, horizon)
        cost = cost_function(final_state)
        candidates.append((cost, control, final_state))
    
    # Sort by cost (best first)
    candidates.sort(key=lambda x: x[0])
    
    # Find first valid (collision-free) candidate
    best_control = candidates[0][1]
    best_state = candidates[0][2]
    for cost, control, final_state in candidates:
        if boat_collision_free(Node(start_state), Node(final_state), obstacles):
            best_control = control
            best_state = final_state
            break

    return best_state, best_control
    