# agent_dynamics.py
"""
Simple kinematic agent dynamics:
The agent moves with constant speed toward a target point.
"""

import numpy as np


class AgentDynamics:
    def __init__(self, x0, speed=0.1):
        """
        Parameters
        ----------
        x0 : np.array
            Initial agent position (2D or 3D).
        speed : float
            Constant linear speed of the agent (units per second).
        """
        self.x = np.asarray(x0, dtype=float)
        self.speed = float(speed)

    def update(self, target, dt):
        """
        Move agent toward the target at constant speed.

        Parameters
        ----------
        target : np.array
            Desired target position.
        dt : float
            Time step.
        """
        target = np.asarray(target, dtype=float)

        # Compute direction vector
        direction = target - self.x
        dist = np.linalg.norm(direction)

        if dist < 1e-6:
            return self.x  # already at target

        # Normalize direction
        direction = direction / dist

        # Distance we can move this frame
        step = self.speed * dt

        # Clamp to avoid overshooting target
        if step >= dist:
            self.x = target.copy()
        else:
            self.x += direction * step

        return self.x
