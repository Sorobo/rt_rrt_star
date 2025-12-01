# Dynamic Steering Implementation

## Overview
Extended the RT-RRT* planner to support 6D state space with boat dynamics.

## State Representation
- **Old**: `[x, y]` or `[x, y, theta]`
- **New**: `[x, y, theta, x_dot, y_dot, theta_dot]`
  - Position: `x, y`
  - Heading: `theta`
  - Velocities: `x_dot, y_dot` (linear), `theta_dot` (angular)

## Key Changes

### 1. Node Module (`node_module.py`)
- Extended to support variable state dimensions (2D, 3D, or 6D)
- Added properties:
  - `position`: Returns `[x, y]`
  - `heading`: Returns `theta`
  - `velocity`: Returns `[x_dot, y_dot, theta_dot]`
- Added `control` field to store control inputs

### 2. Steering (`steer.py`)
**New function: `steer_dynamic(start_state, target_state, dt, horizon)`**
- Uses simplified boat dynamics model
- Optimizes control sequence `[tau_x, tau_y, tau_n]` (forces/torque)
- Minimizes distance to target state
- Returns: optimal state and control input

**Dynamics model:**
```python
# State: [x, y, theta, u, v, r]
# Control: [tau_x, tau_y, tau_n]
# Body frame velocities: u (surge), v (sway), r (yaw rate)
```

### 3. Sampler (`sampler.py`)
- `sample_uniform()` now supports 6D sampling
- Samples reasonable velocity ranges:
  - Linear: ±2.0 m/s
  - Angular: ±0.5 rad/s

### 4. Tree Module (`tree.py`)
- `add_node()` updated to:
  - Use position distance for cost (not full state)
  - Store control input in node
  - Support 6D states

### 5. Configuration (`config.py`)
Added parameters:
```python
USE_DYNAMIC_STEERING = False  # Toggle dynamic steering
DYNAMIC_DT = 0.1  # Time step [s]
DYNAMIC_HORIZON = 10  # Simulation steps
```

## Usage

### Enable Dynamic Steering
1. Set `USE_DYNAMIC_STEERING = True` in `config.py`
2. Ensure initial state is 6D: `x_start = [x, y, theta, 0, 0, 0]`

### Test Dynamic Steering
```bash
python test_dynamic_steering.py
```

This will:
- Compute optimal control from start to target
- Simulate trajectory
- Visualize:
  - XY path
  - Heading evolution
  - Velocity profiles
  - Control inputs

### Integration with RT-RRT*
The system automatically handles both modes:
- **Kinematic** (`USE_DYNAMIC_STEERING=False`): Uses geometric steering
- **Dynamic** (`USE_DYNAMIC_STEERING=True`): Uses dynamics-based steering

## Boat Dynamics Parameters
Based on simplified MilliAmpere1 ferry:
- Mass: 1800 kg
- Yaw inertia: 4860 kg⋅m²
- Damping coefficients: Xu, Yv, Nr
- Max forces: ±400 N (surge/sway)
- Max torque: ±1000 Nm (yaw)

## Advantages of Dynamic Steering
1. **Physically realizable**: Respects dynamics constraints
2. **Velocity aware**: Can plan momentum-aware trajectories
3. **Control generation**: Produces executable control inputs
4. **Better accuracy**: Accounts for inertia and damping

## Next Steps
- Integrate `steer_dynamic()` into main planner
- Add velocity obstacles
- Tune optimization parameters
- Test with full dynamics simulator
