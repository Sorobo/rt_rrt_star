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


def steer_dynamic(start_state, target_state, dt=0.1, horizon=10):
    """
    Dynamic steering using boat dynamics model.
    Tries predefined maximum control inputs and picks the best one.
    
    Parameters
    ----------
    start_state : np.array
        Starting state [x, y, theta, x_dot, y_dot, theta_dot]
    target_state : np.array
        Target state (at least [x, y, theta])
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
    # Ensure we have 6D states
    if len(start_state) < 6:
        start_state = np.pad(start_state, (0, 6 - len(start_state)), 'constant')
    if len(target_state) < 6:
        target_state = np.pad(target_state, (0, 6 - len(target_state)), 'constant')
    
    # Boat parameters (simplified from MilliAmpere1)
    m = 1800.0      # mass [kg]
    Iz = 4860.0     # yaw inertia [kg m^2]
    Xu = -157.0     # linear drag
    Yv = -258.0
    Nr = -1000.0
    
    # Max control limits (maximum power)
    tau_max = 400.0  # Max force in x, y [N]
    tau_n_max = 600.0  # Max torque [Nm]
    
    # Predefined control options - max power in different directions
    control_options = [
        np.array([tau_max, 0.0, 0.0]),           # full forward
        np.array([-tau_max, 0.0, 0.0]),          # full backward
        np.array([0.0, tau_max, 0.0]),           # full right
        np.array([0.0, -tau_max, 0.0]),          # full left
        np.array([0.0, 0.0, tau_n_max]),         # full turn right
        np.array([0.0, 0.0, -tau_n_max]),        # full turn left
        np.array([tau_max, 0.0, tau_n_max]),     # forward + turn right
        np.array([tau_max, 0.0, -tau_n_max]),    # forward + turn left
        np.array([tau_max, tau_max, 0.0]),       # forward + right
        np.array([tau_max, -tau_max, 0.0]),      # forward + left
        np.array([-tau_max, tau_max, 0.0]),      # backward + right
        np.array([-tau_max, -tau_max, 0.0]),     # backward + left
        np.array([0.0, 0.0, 0.0]),               # coast (no control)
    ]
    
    def dynamics(state, control):
        """
        Compute state derivative given current state and control.
        state = [x, y, theta, x_dot, y_dot, theta_dot]
        control = [tau_x, tau_y, tau_n] in body frame
        """
        x, y, theta, u, v, r = state
        tau_x, tau_y, tau_n = control
        
        # Rotation matrix (body to inertial)
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        
        # Damping forces
        damping_x = Xu * u
        damping_y = Yv * v
        damping_n = Nr * r
        
        # Accelerations in body frame
        u_dot = (tau_x + damping_x) / m
        v_dot = (tau_y + damping_y) / m
        r_dot = (tau_n + damping_n) / Iz
        
        # Position derivatives in inertial frame
        x_dot = cos_th * u - sin_th * v
        y_dot = sin_th * u + cos_th * v
        theta_dot = r
        
        return np.array([x_dot, y_dot, theta_dot, u_dot, v_dot, r_dot])
    
    def simulate_forward(state, control):
        """Simulate forward using Euler integration for horizon steps"""
        current_state = state.copy()
        for _ in range(horizon):
            state_dot = dynamics(current_state, control)
            current_state += state_dot * dt
            # Normalize angle
            current_state[2] = np.arctan2(np.sin(current_state[2]), np.cos(current_state[2]))
        return current_state
    
    def cost_function(final_state):
        """Compute cost of reaching a final state"""
        # Position error
        pos_error = np.linalg.norm(final_state[:2] - target_state[:2])
        
        # Heading error
        angle_diff = np.arctan2(np.sin(final_state[2] - target_state[2]), 
                               np.cos(final_state[2] - target_state[2]))
        heading_error = abs(angle_diff)
        
        # Velocity error (prefer matching target velocity if given)
        vel_error = np.linalg.norm(final_state[3:6] - target_state[3:6])
        
        # Weighted cost
        total_cost = pos_error + 0.5 * heading_error + 0.1 * vel_error
        
        return total_cost
    
    # Try all control options and pick the best
    best_cost = float('inf')
    best_control = control_options[0]
    best_state = start_state.copy()
    
    for control in control_options:
        final_state = simulate_forward(start_state, control)
        cost = cost_function(final_state)
        
        if cost < best_cost:
            best_cost = cost
            best_control = control
            best_state = final_state
    
    return best_state, best_control
    