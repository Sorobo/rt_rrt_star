import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import json

# ----------------------------
# Load trajectory and bounding boxes from JSON
# ----------------------------
def load_trajectory_from_json(filepath: str, N_interp: int = None) -> tuple:
    """
    Load trajectory waypoints and convex hull bounding boxes from a JSON file.
    
    Args:
        filepath: Path to the trajectory_export.json file
        N_interp: Number of interpolation points (N+1 total points). If None, uses waypoints as-is.
        
    Returns:
        x_ref: numpy array of shape (N_interp, 6) with [x, y, psi, u, v, r] for each waypoint
        corridors: List of tuples (A, b) representing halfspace constraints for each convex hull
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    trajectory = data.get('trajectory', [])
    convex_hulls = data.get('convex_hulls', [])
    
    # Convert trajectory to numpy array with waypoints
    N_waypoints = len(trajectory)
    waypoints = np.zeros((N_waypoints, 6))
    for i, wp in enumerate(trajectory):
        waypoints[i, 0] = wp['x']
        waypoints[i, 1] = wp['y']
        waypoints[i, 2] = wp['heading']
        # u, v, r are initialized to zero
    
    # Interpolate if N_interp is specified
    if N_interp is not None:
        x_ref = np.zeros((N_interp, 6))
        # Create parameter t from 0 to N_waypoints-1
        t_waypoints = np.arange(N_waypoints)
        t_interp = np.linspace(0, N_waypoints - 1, N_interp)
        
        for dim in range(6):
            x_ref[:, dim] = np.interp(t_interp, t_waypoints, waypoints[:, dim])
    else:
        x_ref = waypoints
    
    # Convert convex hulls to corridor halfspace constraints
    corridors = []
    for hull in convex_hulls:
        vertices = np.array(hull['vertices'])
        A, b = polygon_to_halfspaces(vertices)
        corridors.append((A, b))
    
    return x_ref, corridors


# ----------------------------
# Helper: build constant B from thrusters
# ----------------------------
def build_B(thruster_xy: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """
    thruster_xy: (m,2) array of [x_i, y_i]
    alphas: (m,) fixed angles in radians
    returns B in R^{3 x m}
    """
    m = thruster_xy.shape[0]
    B = np.zeros((3, m))
    for i in range(m):
        x_i, y_i = thruster_xy[i]
        a = alphas[i]
        B[:, i] = np.array([
            np.cos(a),
            np.sin(a),
            x_i*np.sin(a) - y_i*np.cos(a)
        ])
    return B


# ----------------------------
# ASV model builder (CasADi)
# ----------------------------
def build_asv_discrete_model(dt: float, B: np.ndarray, params: dict):
    """
    Builds discrete dynamics x_{k+1} = f(x_k, T_k) using Euler integration.

    State x = [x, y, psi, u, v, r]
    Input T = [T1, T2, T3, T4]
    Tau = B*T  (3x4 * 4x1 => 3x1)

    params must provide:
      m, Iz, Xu_dot, Yv_dot, Yr_dot, Nr_dot, Nv_dot (optional),
      Xu, Yv, Yr, Nv, Nr,
      Xuu, Yvv, Nrr  (quadratic drag coefficients; can be 0)
    """
    # Symbolic variables
    x = ca.SX.sym("x", 6)
    T = ca.SX.sym("T", B.shape[1])

    # Unpack state
    Xp, Yp, psi, u, v, r = x[0], x[1], x[2], x[3], x[4], x[5]
    nu = ca.vertcat(u, v, r)

    # Rotation R(psi) for eta_dot = R * nu
    cpsi, spsi = ca.cos(psi), ca.sin(psi)
    R = ca.SX.zeros(3, 3)
    R[0, 0] = cpsi;  R[0, 1] = -spsi
    R[1, 0] = spsi;  R[1, 1] =  cpsi
    R[2, 2] = 1.0

    # --- Mass matrices (RB + added mass) ---
    m = params["m"]
    Iz = params["Iz"]

    Xu_dot = params.get("Xu_dot", 0.0)
    Yv_dot = params.get("Yv_dot", 0.0)
    Yr_dot = params.get("Yr_dot", 0.0)
    Nv_dot = params.get("Nv_dot", 0.0)  # optional
    Nr_dot = params.get("Nr_dot", 0.0)

    # MRB
    MRB = ca.diag(ca.vertcat(m, m, Iz))

    # MA (your form; adjust if you use a slightly different convention)
    # Provided example looked like:
    # MA = [[-Xu_dot,   0,      0],
    #       [  0,    -Yv_dot, -Yr_dot],
    #       [  0,    -Yr_dot, -Nr_dot]]
    MA = ca.SX.zeros(3, 3)
    MA[0, 0] = -Xu_dot
    MA[1, 1] = -Yv_dot
    MA[1, 2] = -Yr_dot
    MA[2, 1] = -Yr_dot
    MA[2, 2] = -Nr_dot

    M = MRB + MA
    Minv = ca.inv(M)

    # --- Coriolis matrices ---
    # CRB(nu) from your snippet:
    # [[0, 0, -m*v],
    #  [0, 0,  m*u],
    #  [m*v , -m*u, 0]]
    CRB = ca.SX.zeros(3, 3)
    CRB[0, 2] = -m * v
    CRB[1, 2] =  m * u
    CRB[2, 0] =  m * v
    CRB[2, 1] = -m * u

    # CA(nu) (added mass Coriolis). Your snippet suggests:
    # [[0, 0, +Yv_dot*v + Yr_dot*r],
    #  [0, 0,  -Xu_dot*u],
    #  [-Yv_dot*v - Yr_dot*r, +Xu_dot*u, 0]]
    CA = ca.SX.zeros(3, 3)
    CA[0, 2] =  (Yv_dot * v + Yr_dot * r)
    CA[1, 2] = -(Xu_dot * u)
    CA[2, 0] = -(Yv_dot * v + Yr_dot * r)
    CA[2, 1] =  (Xu_dot * u)

    C = CRB + CA

    # --- Damping matrices ---
    # DL: Linear damping matrix
    Xu = params.get("Xu", 0.0)
    Yv = params.get("Yv", 0.0)
    Yr = params.get("Yr", 0.0)
    Nv = params.get("Nv", 0.0)
    Nr = params.get("Nr", 0.0)

    DL = ca.SX.zeros(3, 3)
    DL[0, 0] = -Xu
    DL[1, 1] = -Yv
    DL[1, 2] = -Yr
    DL[2, 1] = -Nv
    DL[2, 2] = -Nr

    # DNL: Nonlinear damping matrix (quadratic drag)
    Xuu = params.get("Xuu", 0.0)  # coefficient for -X|u|u * |u|
    Yvv = params.get("Yvv", 0.0)  # coefficient for -Y|v|v * |v|
    Nrr = params.get("Nrr", 0.0)  # coefficient for -N|r|r * |r|

    # Smooth abs to make derivatives nicer near 0:
    eps_abs = 1e-3
    abs_u = ca.sqrt(u*u + eps_abs*eps_abs)
    abs_v = ca.sqrt(v*v + eps_abs*eps_abs)
    abs_r = ca.sqrt(r*r + eps_abs*eps_abs)

    DNL = ca.SX.zeros(3, 3)
    DNL[0, 0] = -Xuu * abs_u
    DNL[1, 1] = -Yvv * abs_v
    DNL[2, 2] = -Nrr * abs_r

    # Total damping
    D = DL + DNL

    # Input mapping tau = B*T
    tau = ca.DM(B) @ T  # 3x1

    # Continuous dynamics
    eta_dot = R @ nu
    nu_dot = Minv @ (tau - C @ nu - D @ nu)

    # Euler discretization
    eta_next = ca.vertcat(Xp, Yp, psi) + dt * eta_dot
    nu_next  = nu + dt * nu_dot
    x_next = ca.vertcat(eta_next, nu_next)

    f = ca.Function("f", [x, T], [x_next], ["x", "T"], ["xnext"])
    Fx = ca.Function("Fx", [x, T], [ca.jacobian(x_next, x)], ["x", "T"], ["Fx"])
    Fu = ca.Function("Fu", [x, T], [ca.jacobian(x_next, T)], ["x", "T"], ["Fu"])
    return f, Fx, Fu


# ----------------------------
# SCP-QP builder/solver (single horizon solve)
# ----------------------------

def polygon_area_signed(V: np.ndarray) -> float:
    """Signed area. >0 means CCW."""
    x = V[:, 0]
    y = V[:, 1]
    return 0.5 * np.sum(x*np.roll(y, -1) - np.roll(x, -1)*y)

def polygon_to_halfspaces(V: np.ndarray):
    """
    Convert a convex polygon (vertices) to halfspaces A p <= b.
    V: (m,2) vertices, convex, ordered CW or CCW.
    Returns A (m,2), b (m,)
    
    For each edge, we want the halfspace that keeps the interior.
    If polygon is CCW, the inward normal points to the right of each edge.
    """
    V = np.asarray(V, dtype=float)

    # Ensure CCW order
    if polygon_area_signed(V) < 0:
        V = V[::-1].copy()

    m = V.shape[0]
    A = np.zeros((m, 2), dtype=float)
    b = np.zeros((m,), dtype=float)

    for i in range(m):
        v_i = V[i]
        v_j = V[(i + 1) % m]
        e = v_j - v_i  # edge direction vector

        # For CCW polygon, inward normal is the RIGHT perpendicular of edge
        # Right perpendicular of [dx, dy] is [dy, -dx]
        n_in = np.array([e[1], -e[0]], dtype=float)
        
        # Normalize the normal vector for better numerical stability
        norm = np.linalg.norm(n_in)
        if norm > 1e-10:
            n_in = n_in / norm

        # Halfspace: n_in^T * p <= n_in^T * v_i
        # (any point on the inward side satisfies this)
        A[i, :] = n_in
        b[i] = n_in @ v_i

    return A, b

def point_in_poly_halfspace(p: np.ndarray, A: np.ndarray, b: np.ndarray, tol=1e-9) -> bool:
    return np.all(A @ p <= b + tol)


def inside_poly(A, b, p, tol=1e-9):
    return np.all(A @ p <= b + tol)

def build_sigma_monotone(x_ref, corridors, start_idx=0):
    """
    x_ref: (N+1, 6) or (N+1, 3+) - full state with [x, y, psi, ...]
    corridors: list of (A,b)
    start_idx: corridor index at k=0
    
    Assigns each point to a corridor index, advancing monotonically.
    Switches to next corridor as soon as ALL FOUR CORNERS of the boat are inside it.
    """
    # Boat dimensions
    L = 5.0  # boat length [m]
    W = 2.8  # boat width [m]
    
    # Define corners in body frame
    corners_body = np.array([
        [ L/2,  W/2],  # front-right
        [ L/2, -W/2],  # front-left
        [-L/2,  W/2],  # rear-right
        [-L/2, -W/2],  # rear-left
    ])
    
    sigma = np.zeros(x_ref.shape[0], dtype=int)
    i = start_idx
    
    for k in range(x_ref.shape[0]):
        x_k, y_k, psi_k = x_ref[k, 0], x_ref[k, 1], x_ref[k, 2]
        
        # Compute all four corner positions in world frame
        cos_psi = np.cos(psi_k)
        sin_psi = np.sin(psi_k)
        R = np.array([[cos_psi, -sin_psi],
                      [sin_psi,  cos_psi]])
        
        corners_world = np.array([x_k, y_k]) + (R @ corners_body.T).T
        
        # Try to advance to next corridor if ALL corners fit in it
        while i < len(corridors)-1:
            A_next, b_next = corridors[i+1]
            
            # Check if all four corners are inside the next corridor
            all_corners_inside = True
            for corner in corners_world:
                if not inside_poly(A_next, b_next, corner):
                    all_corners_inside = False
                    break
            
            if all_corners_inside:
                # All corners are in next corridor, advance
                i += 1
            else:
                # Not all corners fit in next corridor, stay in current
                break
        
        sigma[k] = i
    
    return sigma

class SCPPlanner:
    def __init__(self, f, Fx, Fu, dt, N, Tmax, Q, R, rho_x, rho_u):
        self.f = f
        self.Fx = Fx
        self.Fu = Fu
        self.dt = dt
        self.N = N
        self.Tmax = Tmax
        self.Q = Q  # 6x6 state weight (for dx)
        self.R = R  # 4x4 input weight (for dT)
        self.rho_x = rho_x
        self.rho_u = rho_u

        # QP solver will be created in solve method with proper structure
        self.qpsolver = None

    def solve(self, x0, x_ref, corridors, sigma, T_init=None, scp_iters=5, alpha=0.3):
        """
        x0: (6,) initial state
        x_ref: (N+1,6) reference trajectory (can be constant or from your waypoint path)
        corridors: list of (A_i, b_i) for each polytope, with A_i shape (m_i,2), b_i shape (m_i,)
        sigma: length N+1 integer array selecting corridor index at each step k
        T_init: optional initial input guess, shape (N,4)
        alpha: step size for SCP updates (default 0.3 for stability)
        """
        N = self.N
        nx = 6
        nu = 4

        # --- initialize nominal trajectories ---
        x_nom = np.zeros((N+1, nx))
        T_nom = np.zeros((N, nu))
        x_nom[0] = x0

        if T_init is not None:
            T_nom[:] = T_init
        else:
            T_nom[:] = 0.0

        # forward simulate nominal
        for k in range(N):
            x_nom[k+1] = np.array(self.f(x_nom[k], T_nom[k]).full()).reshape(-1)

        # SCP loop
        x_nom_prev = None
        T_nom_prev = None
        convergence_tol = 5e-2  # Convergence tolerance
        for it in range(scp_iters):



            sigma = build_sigma_monotone(x_nom, corridors, start_idx=0)
            print(f"SCP Iteration {it}, sigma: {sigma}")
            # Build QP: min 0.5 z'H z + q'z s.t. l <= A z <= u
            # z = [dx_0..dx_N, dT_0..dT_{N-1}]
            n_dx = (N+1) * nx
            n_dT = N * nu
            nz = n_dx + n_dT

            # Helper indexers
            def idx_dx(k):  # returns slice for dx_k
                a = k * nx
                return slice(a, a + nx)

            def idx_dT(k):  # returns slice for dT_k in z
                a = n_dx + k * nu
                return slice(a, a + nu)

            # Quadratic objective: penalize (x_nom+dx - x_ref) and (T_nom+dT)
            H = np.zeros((nz, nz))
            q = np.zeros((nz,))

            for k in range(N+1):
                Qk = self.Q
                # cost on dx: (x_nom+dx - x_ref)^T Q (x_nom+dx - x_ref)
                # => dx^T Q dx + 2 (x_nom - x_ref)^T Q dx + const
                idx_k = idx_dx(k)
                H[idx_k, idx_k] = 2.0 * Qk
                q[idx_k] = 2.0 * (x_nom[k] - x_ref[k]) @ Qk

            for k in range(N):
                Rk = self.R
                idx_k = idx_dT(k)
                H[idx_k, idx_k] = 2.0 * Rk
                q[idx_k] = 2.0 * T_nom[k] @ Rk

            # Constraints:
            # 1) dynamics equalities: dx_{k+1} - F dx_k - G dT_k = c
            # 2) corridor inequalities: A (p_nom + dp) <= b
            # 3) input box: -Tmax <= T_nom + dT <= Tmax
            # 4) trust region: |dx|_inf <= rho_x, |dT|_inf <= rho_u

            A_rows = []
            l_rows = []
            u_rows = []

            # Initial condition: dx_0 = x0 - x_nom0 (but x_nom0 == x0, so 0)
            # We'll force dx_0 = 0 for stability.
            A0 = np.zeros((nx, nz))
            A0[:, idx_dx(0)] = np.eye(nx)
            A_rows.append(A0)
            l_rows.append(np.zeros(nx))
            u_rows.append(np.zeros(nx))

            # Dynamics constraints
            for k in range(N):
                Fxk = np.array(self.Fx(x_nom[k], T_nom[k]).full())
                Fuk = np.array(self.Fu(x_nom[k], T_nom[k]).full())
                fk  = np.array(self.f(x_nom[k], T_nom[k]).full()).reshape(-1)

                # c = f(x_nom, T_nom) - x_nom_next
                ck = fk - x_nom[k+1]

                Aeq = np.zeros((nx, nz))
                Aeq[:, idx_dx(k+1)] = np.eye(nx)
                Aeq[:, idx_dx(k)]   = -Fxk
                Aeq[:, idx_dT(k)]   = -Fuk

                A_rows.append(Aeq)
                l_rows.append(ck)  # equality
                u_rows.append(ck)

            # Corridor inequalities: A p <= b
            # For a boat with length L=5m and width W=2.8m, check all 4 corners
            L = 5.0  # boat length [m]
            W = 2.8  # boat width [m]
            
            # Define corners in body frame (relative to center)
            corners_body = np.array([
                [ L/2,  W/2],  # front-right
                [ L/2, -W/2],  # front-left
                [-L/2,  W/2],  # rear-right
                [-L/2, -W/2],  # rear-left
            ])
            
            for k in range(N+1):
                poly_idx = int(sigma[k])
                A2, b2 = corridors[poly_idx]
                A2 = np.asarray(A2)
                b2 = np.asarray(b2).reshape(-1)
                mfaces = A2.shape[0]
                
                # Get state at time k
                x_k, y_k, psi_k = x_nom[k, 0], x_nom[k, 1], x_nom[k, 2]
                
                # For each corner, add constraints
                for corner in corners_body:
                    # Rotate corner by heading angle and translate to world frame
                    cos_psi = np.cos(psi_k)
                    sin_psi = np.sin(psi_k)
                    
                    # Rotation matrix
                    R = np.array([[cos_psi, -sin_psi],
                                  [sin_psi,  cos_psi]])
                    
                    # Corner position in world frame (nominal)
                    corner_world_nom = np.array([x_k, y_k]) + R @ corner
                    
                    # Linearization: p_corner ≈ p_corner_nom + J * dx
                    # where J is the Jacobian of corner position w.r.t. state
                    # dp_corner/dx = [1, 0], dp_corner/dy = [0, 1]
                    # dp_corner/dpsi = [-corner_x*sin(psi) - corner_y*cos(psi),
                    #                    corner_x*cos(psi) - corner_y*sin(psi)]
                    
                    J_psi_x = -corner[0]*sin_psi - corner[1]*cos_psi
                    J_psi_y =  corner[0]*cos_psi - corner[1]*sin_psi
                    
                    # Constraint: A2 * p_corner <= b2
                    # A2 * (p_corner_nom + J*dx) <= b2
                    # A2 * J * dx <= b2 - A2 * p_corner_nom
                    
                    Aineq = np.zeros((mfaces, nz))
                    # J = [[1, 0, J_psi_x],
                    #      [0, 1, J_psi_y]]  (2x3 for x, y, psi)
                    # A2 * J gives (mfaces x 3)
                    
                    Aineq[:, idx_dx(k).start + 0] = A2[:, 0]  # dx
                    Aineq[:, idx_dx(k).start + 1] = A2[:, 1]  # dy
                    Aineq[:, idx_dx(k).start + 2] = A2[:, 0] * J_psi_x + A2[:, 1] * J_psi_y  # dpsi
                    
                    rhs = b2 - A2 @ corner_world_nom
                    A_rows.append(Aineq)
                    l_rows.append(-np.inf * np.ones(mfaces))
                    u_rows.append(rhs)

            # Input saturation: -Tmax <= T_nom + dT <= Tmax => -Tmax - T_nom <= dT <= Tmax - T_nom
            for k in range(N):
                Abox = np.zeros((nu, nz))
                Abox[:, idx_dT(k)] = np.eye(nu)
                l = -self.Tmax - T_nom[k]
                u =  self.Tmax - T_nom[k]
                A_rows.append(Abox)
                l_rows.append(l)
                u_rows.append(u)

            # Trust region: |dx|_inf <= rho_x  and |dT|_inf <= rho_u (box constraints)
            # Implemented as simple bounds on variables via extra identity constraints
            # dx bounds
            Adx = np.zeros((n_dx, nz))
            Adx[:, :n_dx] = np.eye(n_dx)
            A_rows.append(Adx)
            l_rows.append(-self.rho_x * np.ones(n_dx))
            u_rows.append( self.rho_x * np.ones(n_dx))

            # dT bounds
            AdT = np.zeros((n_dT, nz))
            AdT[:, n_dx:] = np.eye(n_dT)
            A_rows.append(AdT)
            l_rows.append(-self.rho_u * np.ones(n_dT))
            u_rows.append( self.rho_u * np.ones(n_dT))

            # Stack constraints
            A = np.vstack(A_rows)
            l = np.concatenate(l_rows)
            u = np.concatenate(u_rows)

            # Solve QP using CasADi
            # Define problem structure
            x_var = ca.SX.sym('x', nz)
            qp_prob = {
                'x': x_var,
                'f': 0.5 * ca.mtimes([x_var.T, ca.DM(H), x_var]) + ca.dot(ca.DM(q), x_var),
                'g': ca.mtimes(ca.DM(A), x_var)
            }
            
            # Create solver with relaxed options
            opts = {
                'error_on_fail': False,
                'osqp': {
                    'verbose': False,
                    'polish': True,
                    'max_iter': 10000,
                    'eps_abs': 1e-3,
                    'eps_rel': 1e-3,
                    'eps_prim_inf': 1e-4,
                    'eps_dual_inf': 1e-4
                }
            }
            qpsolver = ca.qpsol('qp_iter_' + str(it), 'osqp', qp_prob, opts)
            
            # Solve
            sol = qpsolver(lbg=ca.DM(l), ubg=ca.DM(u))
            
            # Check solver status
            stats = qpsolver.stats()
            return_status = stats.get('return_status', 'unknown')
            
            # Accept solved or solved_inaccurate as valid
            if return_status not in ['solved', 'solved inaccurate']:
                print(f"Warning: QP solver failed at iteration {it}. Return code: {return_status}")
                print(f"Stats: {stats}")
                break
            
            if return_status == 'solved inaccurate':
                print(f"Iteration {it}: solved with reduced accuracy")
            
            z = np.array(sol["x"]).reshape(-1)

            # Check for NaN
            if np.any(np.isnan(z)):
                print(f"Warning: NaN detected in solution at iteration {it}")
                break

            # Extract updates
            dx = z[:n_dx].reshape((N+1, nx))
            dT = z[n_dx:].reshape((N, nu))

            # Update nominal with reduced step size for stability
            x_nom_new = x_nom + alpha * dx
            T_nom_new = T_nom + alpha * dT
            
            # Check for NaN in updated values
            if np.any(np.isnan(x_nom_new)) or np.any(np.isnan(T_nom_new)):
                print(f"Warning: NaN detected after update at iteration {it}")
                break
            
            # Check for convergence
            if x_nom_prev is not None and T_nom_prev is not None:
                x_change = np.linalg.norm(x_nom_new - x_nom_prev) / (np.linalg.norm(x_nom_prev) + 1e-10)
                T_change = np.linalg.norm(T_nom_new - T_nom_prev) / (np.linalg.norm(T_nom_prev) + 1e-10)
                
                if x_change < convergence_tol and T_change < convergence_tol:
                    print(f"Converged at iteration {it}: x_change={x_change:.2e}, T_change={T_change:.2e}")
                    x_nom = x_nom_new
                    T_nom = T_nom_new
                    print("aborted because of convergence, at iteration:", it)
                    break
            
            # Store previous values for next convergence check
            x_nom_prev = x_nom.copy()
            T_nom_prev = T_nom.copy()
                
            x_nom = x_nom_new
            T_nom = T_nom_new

            # Re-simulate nominal to keep consistency (optional but often stabilizes SCP)
            x_nom[0] = x0
            nan_detected = False
            for k in range(N):
                xk_next = np.array(self.f(x_nom[k], T_nom[k]).full()).reshape(-1)
                if np.any(np.isnan(xk_next)) or np.any(np.isinf(xk_next)):
                    print(f"Warning: NaN/Inf in dynamics at iteration {it}, step {k}")
                    print(f"x_nom[{k}] = {x_nom[k]}")
                    print(f"T_nom[{k}] = {T_nom[k]}")
                    nan_detected = True
                    break
                x_nom[k+1] = xk_next
            
            if nan_detected:
                print("Stopping SCP due to numerical instability")
                break

        return x_nom, T_nom


# ----------------------------
# Plotting function
# ----------------------------
def plot_results(x_plan, T_plan, x_ref, dt, Tmax, corridors=None):
    """
    Plot the trajectory, velocities, and thrust inputs.
    
    x_plan: (N+1, 6) array of states
    T_plan: (N, 4) array of thrust inputs
    x_ref: (N+1, 6) reference trajectory
    dt: time step
    Tmax: maximum thrust
    corridors: list of (A, b) corridor constraints (optional)
    """
    N = T_plan.shape[0]
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. X-Y trajectory plot
    ax1 = plt.subplot(3, 3, 1)
    
    # Draw corridor if provided
    if corridors is not None:
        for A, b in corridors:
            # Convert halfspace representation A*p <= b to vertices for plotting
            from matplotlib.patches import Polygon
            from scipy.spatial import HalfspaceIntersection
            
            try:
                # Create halfspaces in format [A | -b], i.e., A*x + b_offset <= 0
                # Our format is A*p <= b, so we need A*p - b <= 0
                halfspaces = np.hstack([A, -b.reshape(-1, 1)])
                
                # Find an interior point (centroid estimate)
                # Try solving the LP to find a feasible point
                interior_point = None
                
                # Simple heuristic: average of points that satisfy all constraints
                grid_x = np.linspace(-20, 20, 20)
                grid_y = np.linspace(-20, 20, 20)
                feasible_points = []
                for gx in grid_x:
                    for gy in grid_y:
                        p = np.array([gx, gy])
                        if np.all(A @ p <= b + 1e-6):
                            feasible_points.append(p)
                
                if len(feasible_points) > 0:
                    interior_point = np.mean(feasible_points, axis=0)
                    
                    # Use HalfspaceIntersection to find vertices
                    hs = HalfspaceIntersection(halfspaces, interior_point)
                    vertices = hs.intersections
                    
                    # Sort vertices by angle for proper polygon plotting
                    center = np.mean(vertices, axis=0)
                    angles = np.arctan2(vertices[:, 1] - center[1], 
                                       vertices[:, 0] - center[0])
                    sorted_idx = np.argsort(angles)
                    vertices_sorted = vertices[sorted_idx]
                    
                    corridor_poly = Polygon(vertices_sorted, fill=True, alpha=0.2, 
                                          facecolor='green', edgecolor='green', 
                                          linewidth=2, label='Corridor')
                    ax1.add_patch(corridor_poly)
            except Exception as e:
                print(f"Warning: Could not plot corridor: {e}")
                # Fallback: just don't plot it
                pass
    
    # Draw boat at several positions along trajectory
    L = 5.0  # boat length
    W = 2.8  # boat width
    boat_corners_body = np.array([
        [ L/2,  W/2],  # front-right
        [ L/2, -W/2],  # front-left
        [-L/2, -W/2],  # rear-left
        [-L/2,  W/2],  # rear-right
        [ L/2,  W/2],  # close the polygon
    ])
    
    # Plot boat at start, middle, and end positions
    plot_indices = [0, N//2, N]
    colors = ['green', 'blue', 'red']
    labels = ['Start', 'Middle', 'End']
    
    for idx, color, label in zip(plot_indices, colors, labels):
        x_k, y_k, psi_k = x_plan[idx, 0], x_plan[idx, 1], x_plan[idx, 2]
        
        # Rotation matrix
        cos_psi = np.cos(psi_k)
        sin_psi = np.sin(psi_k)
        R = np.array([[cos_psi, -sin_psi],
                      [sin_psi,  cos_psi]])
        
        # Transform boat corners to world frame
        boat_corners_world = np.array([x_k, y_k]) + (R @ boat_corners_body.T).T
        
        from matplotlib.patches import Polygon
        boat_poly = Polygon(boat_corners_world, fill=False, edgecolor=color, 
                           linewidth=2, linestyle='-', label=f'Boat {label}')
        ax1.add_patch(boat_poly)
    
    ax1.plot(x_plan[:, 0], x_plan[:, 1], 'b-', linewidth=2, label='Trajectory')
    ax1.plot(x_plan[0, 0], x_plan[0, 1], 'go', markersize=8)
    ax1.plot(x_plan[-1, 0], x_plan[-1, 1], 'ro', markersize=8)
    ax1.plot(x_ref[:, 0], x_ref[:, 1], 'r--', alpha=0.5, label='Reference')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Vessel Trajectory')
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')
    
    # 2. X vs Time
    ax2 = plt.subplot(3, 3, 2)
    time = np.arange(N+1) * dt
    ax2.plot(time, x_plan[:, 0], 'b-', linewidth=2, label='Actual')
    ax2.plot(time, x_ref[:, 0], 'r--', linewidth=2, alpha=0.5, label='Reference')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('X [m]')
    ax2.set_title('X Position vs Time')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Y vs Time
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(time, x_plan[:, 1], 'b-', linewidth=2, label='Actual')
    ax3.plot(time, x_ref[:, 1], 'r--', linewidth=2, alpha=0.5, label='Reference')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Y [m]')
    ax3.set_title('Y Position vs Time')
    ax3.grid(True)
    ax3.legend()
    
    # 4. Heading angle
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(time, np.rad2deg(x_plan[:, 2]), 'b-', linewidth=2)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Heading [deg]')
    ax4.set_title('Heading Angle')
    ax4.grid(True)
    
    # 5. Surge velocity (u)
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(time, x_plan[:, 3], 'b-', linewidth=2, label='u')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('u [m/s]')
    ax5.set_title('Surge Velocity')
    ax5.grid(True)
    ax5.legend()
    
    # 6. Sway velocity (v)
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(time, x_plan[:, 4], 'b-', linewidth=2, label='v')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('v [m/s]')
    ax6.set_title('Sway Velocity')
    ax6.grid(True)
    ax6.legend()
    
    # 7. Yaw rate (r)
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(time, np.rad2deg(x_plan[:, 5]), 'b-', linewidth=2, label='r')
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('r [deg/s]')
    ax7.set_title('Yaw Rate')
    ax7.grid(True)
    ax7.legend()
    
    # 8. Thrust inputs
    ax8 = plt.subplot(3, 3, 8)
    time_input = np.arange(N) * dt
    ax8.plot(time_input, T_plan[:, 0], 'r-', linewidth=2, label='T1')
    ax8.plot(time_input, T_plan[:, 1], 'g-', linewidth=2, label='T2')
    ax8.plot(time_input, T_plan[:, 2], 'b-', linewidth=2, label='T3')
    ax8.plot(time_input, T_plan[:, 3], 'm-', linewidth=2, label='T4')
    ax8.axhline(y=Tmax, color='k', linestyle='--', alpha=0.5, label='Limits')
    ax8.axhline(y=-Tmax, color='k', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('Thrust [N]')
    ax8.set_title('Thruster Inputs')
    ax8.grid(True)
    ax8.legend()
    
    plt.tight_layout()
    plt.show()


# ----------------------------
# Example usage with your thrusters
# ----------------------------
if __name__ == "__main__":
    thrusters = np.array([
        [ 1.5,  1.0],   # front-right
        [ 1.5, -1.0],   # front-left
        [-1.5,  1.0],   # rear-right
        [-1.5, -1.0],   # rear-left
    ])
    alphas = np.deg2rad([45, -45, 135, -135])
    Tmax = 403

    B = build_B(thrusters, alphas)

    dt = 1
    N = 40

    # Put your actual vessel params here (these are placeholders!)
    params = dict(
        m=1800.0,
        Iz=4860.0,
        Xu_dot= -215.0,
        Yv_dot= -1252.0,
        Yr_dot= -183.0,
        Nr_dot= -3500.0,
        Xu=-157.0,  Yv=-258.0, Yr=147,
        Nv=100.0,   Nr=-1000.0,
        Xuu=-234,  Yvv=-389, Nrr=-1000.0,
    )

    f, Fx, Fu = build_asv_discrete_model(dt, B, params)

    # Weights
    Q = np.diag([150, 150, 5,  1, 1, 1])   # penalize position strongly
    Rw = np.diag([1e-4, 1e-4, 1e-4, 1e-4])
    #Rw = np.diag([0,0,0,0])

    # Increase trust region bounds to allow reaching distant goals
    planner = SCPPlanner(f, Fx, Fu, dt, N, Tmax, Q, Rw, rho_x=2.0, rho_u=50.0)

    x0 = np.array([5, 5, 0, 0, 0, 0], dtype=float)


    # Define the end state
    x_end = np.array([20, 20, 0, 0, 0, 0], dtype=float)

    # Linear interpolation between x0 and x_end for reference trajectory

    x_ref, corridors = load_trajectory_from_json("trajectory_export.json", N_interp=N+1)
    sigma = build_sigma_monotone(x_ref, corridors, start_idx=0)
    x0 = x_ref[0]
    print("sigma:", sigma)
    # Increase SCP iterations for better convergence
    x_plan, T_plan = planner.solve(x0, x_ref, corridors, sigma, scp_iters=60)

    print("First control:", T_plan[0])
    print("Final state:", x_plan[-1])

    # Plot results
    plot_results(x_plan, T_plan, x_ref, dt, Tmax, corridors)
