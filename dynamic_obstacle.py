# dynamic_obstacle.py

import numpy as np

class DynamicObstacle:
    """
    Represents a moving obstacle with position, radius, and velocity.
    """
    def __init__(self, center, radius, velocity):
        """
        Args:
            center: [x, y] position
            radius: obstacle radius
            velocity: [vx, vy] velocity in units per second
        """
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.velocity = np.array(velocity, dtype=float)
    
    def update(self, dt):
        """Update position based on velocity and time delta"""
        self.center += self.velocity * dt
    
    def get_tuple(self):
        """Return (center, radius) tuple for collision checking"""
        return (self.center, self.radius)
    
    def bounce_in_bounds(self, bounds):
        """Bounce off boundaries by reversing velocity"""
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        
        if self.center[0] - self.radius < x_min or self.center[0] + self.radius > x_max:
            self.velocity[0] *= -1
            # Clamp position
            self.center[0] = np.clip(self.center[0], x_min + self.radius, x_max - self.radius)
        
        if self.center[1] - self.radius < y_min or self.center[1] + self.radius > y_max:
            self.velocity[1] *= -1
            # Clamp position
            self.center[1] = np.clip(self.center[1], y_min + self.radius, y_max - self.radius)
