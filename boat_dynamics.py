from math import tau
import numpy as np
from scipy.optimize import minimize

from config import BOAT_LENGTH, BOAT_WIDTH

class MilliAmpere1Sim:
    """
    Simplified milliAmpere1 autonomous ferry simulator.
    Implements 3-DOF vessel dynamics, PID-based DP control,
    and thrust allocation (QP minimization).

    Reference:
    Hinostroza et al., "Model Identification, Dynamic Positioning, 
    and Thrust Allocation System for the milliAmpere1 Autonomous Ferry Prototype", IEEE Access, 2025.
    """

    def __init__(self,start_pos, dt=0.2):
        # --- Simulation time step ---
        self.dt = dt

        # --- Physical parameters ---
        self.m = 1800.0         # mass [kg]
        self.Iz = 4860.0        # yaw inertia [kg m^2]
        self.Xu_dot = -215.0
        self.Yv_dot = -1252.0
        self.Nr_dot = -3500.0
        self.Xu = -157.0
        self.Yv = -258.0
        self.Yr = 147.0
        self.Nv = 100.0
        self.Nr = -1000.0
        self.Xuu = -234.0
        self.Yvv = -389.0
        self.Nrr = -1000.0
        
        self.length = BOAT_LENGTH
        self.width = BOAT_WIDTH
        
        

        # --- Thruster configuration ---
        self.thrusters = np.array([
            [ 1.5,  1.0],   # front-right
            [ 1.5, -1.0],   # front-left
            [-1.5,  1.0],   # rear-right
            [-1.5, -1.0]    # rear-left
        ])
        self.alphas = np.deg2rad([45, -45, 135, -135])  # fixed angles
        self.Tmax = 100.0                               # N per thruster

        # --- Controller gains ---
        self.Kp = np.diag([400, 400, 400])
        self.Ki = np.diag([0.01, 0.01, 0.01])
        self.Kd = np.diag([1200, 1200, 500])

        # --- State vectors ---
        self.x = np.zeros(3)#start_pos  # [x, y, psi]
        self.x[0] = start_pos[0]
        self.x[1] = start_pos[1]
        self.nu = np.zeros(3)   # [u, v, r]
        self.int_err = np.zeros(3)

        # --- Data logging ---
        self.traj_eta = []
        self.traj_ref = []

    # ------------------------------------------------------------------
    # --- Core vessel dynamics ---
    # ------------------------------------------------------------------
    @staticmethod
    def rotation(psi):
        """Rotation matrix from body to inertial frame."""
        return np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi),  np.cos(psi), 0],
            [0, 0, 1]
        ])

    def get_corners(self):
        """Return the four corners of the boat in the inertial frame."""
        # Boat dimensions (half-lengths from center)
        L_half = self.length / 2  # half length (front/rear thrusters are at ±1.5)
        W_half = self.width / 2  # half width (thrusters are at ±1.0)
        
        # Corner positions in body frame
        corners_body = np.array([
            [ L_half,  W_half],  # front-right
            [ L_half, -W_half],  # front-left
            [-L_half, -W_half],  # rear-left
            [-L_half,  W_half]   # rear-right
        ])
        
        # Transform to inertial frame
        psi = self.x[2]
        R = np.array([
            [np.cos(psi), -np.sin(psi)],
            [np.sin(psi),  np.cos(psi)]
        ])
        
        corners_inertial = (R @ corners_body.T).T + self.x[:2]
        return corners_inertial
    def damping(self, v):
        """Linear damping force."""
        return np.diag([-self.Xu, -self.Yv, -self.Nr]) @ v

    def coriolis(self, v):
        """Rigid-body Coriolis and centripetal matrix."""
        u, v_, r = v
        m = self.m
        return np.array([
            [0, 0, -m*v_],
            [0, 0,  m*u],
            [m*v_, -m*u, 0]
        ]) @ v

    def system_dynamics(self, eta, nu, tau):
        """3-DOF vessel equations of motion."""
        M = np.diag([
            self.m - self.Xu_dot,
            self.m - self.Yv_dot,
            self.Iz - self.Nr_dot
        ])
        nu_dot = np.linalg.inv(M) @ (tau - self.coriolis(nu) - self.damping(nu))
        eta_dot = self.rotation(eta[2]) @ nu
        return eta_dot, nu_dot

    # ------------------------------------------------------------------
    # --- Thrust allocation (Quadratic Programming) ---
    # ------------------------------------------------------------------
    def thrust_allocation(self, tau_c):
        """Solve for individual thruster thrusts (QP simplified to least-squares)."""
        B = np.zeros((3, 4))
        for i, (xy, a) in enumerate(zip(self.thrusters, self.alphas)):
            x, y = xy
            B[:, i] = [np.cos(a), np.sin(a), x*np.sin(a) - y*np.cos(a)]

        def cost(T): return np.sum((T / self.Tmax) ** 2)
        def cons(T): return B @ T - tau_c

        res = minimize(cost, np.zeros(4),
                       constraints={'type': 'eq', 'fun': cons},
                       bounds=[(0, self.Tmax)] * 4)
        return res.x if res.success else np.zeros(4)

    # ------------------------------------------------------------------
    # --- Control law ---
    # ------------------------------------------------------------------
    def dp_controller(self, eta_ref):
        """Compute desired forces using model-based PID."""
        e = self.rotation(self.x[2]).T @ (eta_ref - self.x)
        e[2] = np.arctan2(np.sin(e[2]), np.cos(e[2]))  # wrap angle

        self.int_err += e * self.dt
        nu_ref = np.zeros(3)
        nu_e = nu_ref - self.nu

        tau_fb = self.Kp @ e + self.Ki @ self.int_err + self.Kd @ nu_e
        tau_ff = np.zeros(3)  # no feedforward term in this simplified version
        return tau_ff + tau_fb

    # ------------------------------------------------------------------
    # --- Simulation step ---
    # ------------------------------------------------------------------
    def step(self, eta_ref):
        """One integration step of simulation."""
        tau_cmd = self.dp_controller(eta_ref)
        #Ti = self.thrust_allocation(tau_cmd)

        # Compute total forces from allocated thrusts
        tau_thr = np.zeros(3)
        """
        for (xy, a, T) in zip(self.thrusters, self.alphas, Ti):
            x, y = xy
            tau_thr += np.array([
                T * np.cos(a),
                T * np.sin(a),
                x * T * np.sin(a) - y * T * np.cos(a)
            ])
        """
     
        tau_thr[0] = np.clip(tau_cmd[0], -4*self.Tmax, 4*self.Tmax)
        tau_thr[1] = np.clip(tau_cmd[1], -4*self.Tmax, 4*self.Tmax)
        tau_thr[2] = np.clip(tau_cmd[2], -40*self.Tmax, 40*self.Tmax)

        
        # Integrate dynamics
        eta_dot, nu_dot = self.system_dynamics(self.x, self.nu, tau_thr)
        self.x += eta_dot * self.dt
        self.nu += nu_dot * self.dt

        # Log data
        self.traj_eta.append(self.x.copy())
        self.traj_ref.append(eta_ref.copy())

    # ------------------------------------------------------------------
    # --- Run complete simulation ---
    # ------------------------------------------------------------------
    def run(self, eta_ref, t_end=10.0, trajectory=None):
        steps = int(t_end / self.dt)
        #i_traj = 0

        for pose in trajectory:
            eta_ref = pose[1:4]
            self.step(eta_ref)
        eta_ref = trajectory[-1][1:4]
        end_time = 20
        end_steps = int(end_time / self.dt)
        for _ in range(end_steps):
            self.step(eta_ref)
        return np.array(self.traj_eta), np.array(self.traj_ref)
    

    def run_dist(self,trajectory,lead_dist,end_dist):

        eta_ref = trajectory[0][1:4]
        
        i = 0
        while True:
            dist = np.linalg.norm(self.x[:2] - eta_ref[:2])
            if dist < lead_dist:
                i +=1
            if i >= len(trajectory):
                break
            eta_ref = trajectory[i][1:4]
            
            self.step(eta_ref)
        eta_ref = trajectory[-1][1:4]
        while np.linalg.norm(self.x[:2] - eta_ref[:2]) > end_dist:
            self.step(eta_ref)
        return np.array(self.traj_eta), np.array(self.traj_ref)
    
    # ------------------------------------------------------------------
    # --- Methods for dynamic steering ---
    # ------------------------------------------------------------------
    def simulate_with_control(self, state_6d, control, dt, horizon):
        """
        Simulate forward from a 6D state with constant control input.
        
        Parameters
        ----------
        state_6d : np.array
            [x, y, theta, u, v, r] - position, heading, velocities
        control : np.array
            [tau_x, tau_y, tau_n] - forces and torque in body frame
        dt : float
            Time step
        horizon : int
            Number of steps to simulate
        
        Returns
        -------
        final_state : np.array
            Final 6D state after simulation
        """
        # Set state without modifying the object's state
        eta = state_6d[:3].copy()  # [x, y, theta]
        nu = state_6d[3:6].copy()  # [u, v, r]
        
        tau = control.copy()
        
        for _ in range(horizon):
            eta_dot, nu_dot = self.system_dynamics(eta, nu, tau)
            eta += eta_dot * dt
            nu += nu_dot * dt
            # Normalize angle
            eta[2] = np.arctan2(np.sin(eta[2]), np.cos(eta[2]))
        
        return np.concatenate([eta, nu])
    
    @staticmethod
    def get_max_control_options():
        """
        Get predefined maximum control options for steering.
        
        Returns
        -------
        control_options : list of np.array
            List of control vectors [tau_x, tau_y, tau_n]
        """
        tau_max = 200.0  # Max force in x, y [N]
        tau_n_max = 6000.0  # Max torque [Nm]
        
        return [
            np.array([tau_max, 0.0, 0.0]),           # full forward
            np.array([-tau_max, 0.0, 0.0]),          # full backward
            np.array([0.0, tau_max, 0.0]),           # full right
            np.array([0.0, -tau_max, 0.0]),          # full left
            np.array([0.0, 0.0, tau_n_max]),         # full turn right
            np.array([0.0, 0.0, -tau_n_max]),        # full turn left
            np.array([tau_max, 0.0, tau_n_max]),     # forward + turn right
            np.array([tau_max, 0.0, -tau_n_max]), 
            np.array([0, tau_max, -tau_n_max]),      # forward + turn left
            np.array([0, -tau_max, tau_n_max]),      # backward + turn right
            np.array([tau_max, tau_max, 0.0]),       # forward + right
            np.array([tau_max, -tau_max, 0.0]),      # forward + left
            np.array([-tau_max, tau_max, 0.0]),      # backward + right
            np.array([-tau_max, -tau_max, 0.0]),     # backward + left
            np.array([0.0, 0.0, 0.0]),               # coast (no control)
        ]
            
