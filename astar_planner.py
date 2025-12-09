"""
A* path planner for boat with heading-aware grid search.
Uses discretized state space (x, y, heading) with motion primitives.
Supports real-time operation with time budgets and incremental planning.
"""

import numpy as np
import heapq
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from collision import boat_path_collision_free
from config import WORLD_BOUNDS, BOAT_WIDTH, BOAT_LENGTH
import astar_config as cfg


class PlannerStatus(Enum):
    """Status of the planner."""
    NOT_STARTED = "not_started"
    PLANNING = "planning"
    PATH_FOUND = "path_found"
    NO_PATH = "no_path"
    TIME_EXPIRED = "time_expired"


@dataclass(order=True)
class AStarNode:
    """Node for A* search with priority queue ordering."""
    f: float  # Total cost (g + h)
    g: float = field(compare=False)  # Cost from start
    h: float = field(compare=False)  # Heuristic to goal
    x: float = field(compare=False)  # X position (meters)
    y: float = field(compare=False)  # Y position (meters)
    heading: float = field(compare=False)  # Heading (radians)
    parent: Optional['AStarNode'] = field(default=None, compare=False)
    
    @property
    def grid_pos(self) -> Tuple[int, int, int]:
        """Get discretized grid position."""
        grid_x = int(np.round(self.x / cfg.GRID_RESOLUTION))
        grid_y = int(np.round(self.y / cfg.GRID_RESOLUTION))
        # Wrap heading to [0, 2π) and discretize
        heading_wrapped = self.heading % (2 * np.pi)
        grid_heading = int(np.round(heading_wrapped / cfg.HEADING_RESOLUTION)) % int(2 * np.pi / cfg.HEADING_RESOLUTION)
        return (grid_x, grid_y, grid_heading)
    
    @property
    def state(self) -> np.ndarray:
        """Get state as numpy array."""
        return np.array([self.x, self.y, self.heading])


class AStarPlanner:
    """A* path planner with heading-aware grid search and real-time capabilities."""
    
    def __init__(self):
        """Initialize A* planner."""
        self.grid_resolution = cfg.GRID_RESOLUTION
        self.heading_resolution = cfg.HEADING_RESOLUTION
        self.motion_primitives = cfg.MOTION_PRIMITIVES
        
        # Planning state for incremental operation
        self.status = PlannerStatus.NOT_STARTED
        self.open_set = []
        self.closed_set = set()
        self.g_score = {}
        self.start_node = None
        self.goal = None
        self.obstacles = None
        self.iterations = 0
        self.nodes_expanded = 0
        self.best_node = None  # Best node found so far (closest to goal)
        
        # Visualization data
        self.closed_nodes = []  # List of closed nodes for visualization
        self.open_nodes = []    # List of open nodes for visualization
        self.current_node = None  # Currently expanding node
        
    def _heuristic(self, node: AStarNode, goal: np.ndarray) -> float:
        """
        Compute heuristic cost to goal.
        Uses Euclidean distance plus heading difference.
        """
        # Euclidean distance
        pos_dist = np.sqrt((node.x - goal[0])**2 + (node.y - goal[1])**2)
        
        # Heading difference (shortest angular distance)
        heading_diff = np.abs(np.arctan2(
            np.sin(node.heading - goal[2]), 
            np.cos(node.heading - goal[2])
        ))
        
        return cfg.HEURISTIC_WEIGHT * (pos_dist + cfg.HEADING_WEIGHT * heading_diff)
    
    def _is_goal_reached(self, node: AStarNode, goal: np.ndarray) -> bool:
        """Check if node is close enough to goal."""
        pos_dist = np.sqrt((node.x - goal[0])**2 + (node.y - goal[1])**2)
        heading_diff = np.abs(np.arctan2(
            np.sin(node.heading - goal[2]), 
            np.cos(node.heading - goal[2])
        ))
        
        return (pos_dist < cfg.GOAL_TOLERANCE_XY and 
                heading_diff < cfg.GOAL_TOLERANCE_HEADING)
    
    def _apply_motion_primitive(self, node: AStarNode, 
                                forward_dist: float, 
                                turn_angle: float) -> Tuple[float, float, float]:
        """
        Apply motion primitive to get new state.
        Uses simple arc motion model.
        """
        current_heading = node.heading
        
        if abs(turn_angle) < 1e-6:
            # Straight line motion
            new_x = node.x + forward_dist * np.cos(current_heading)
            new_y = node.y + forward_dist * np.sin(current_heading)
            new_heading = current_heading
        else:
            # Arc motion
            # Radius of turn
            radius = forward_dist / turn_angle
            
            # Center of arc
            cx = node.x - radius * np.sin(current_heading)
            cy = node.y + radius * np.cos(current_heading)
            
            # New position
            new_heading = current_heading + turn_angle
            new_x = cx + radius * np.sin(new_heading)
            new_y = cy - radius * np.cos(new_heading)
        
        # Normalize heading to [-π, π]
        new_heading = np.arctan2(np.sin(new_heading), np.cos(new_heading))
        
        return new_x, new_y, new_heading
    
    def _check_path_collision(self, node1: AStarNode, node2: AStarNode,
                             obstacles: List[Tuple[np.ndarray, float]]) -> bool:
        """Check if path between two nodes is collision-free."""
        return boat_path_collision_free(
            np.array([node1.x, node1.y]), node1.heading,
            np.array([node2.x, node2.y]), node2.heading,
            BOAT_WIDTH, BOAT_LENGTH,
            obstacles
        )
    
    def _in_bounds(self, x: float, y: float) -> bool:
        """Check if position is within world bounds."""
        return (WORLD_BOUNDS[0, 0] <= x <= WORLD_BOUNDS[0, 1] and
                WORLD_BOUNDS[1, 0] <= y <= WORLD_BOUNDS[1, 1])
    
    def initialize_planning(self, start: np.ndarray, goal: np.ndarray,
                           obstacles: List[Tuple[np.ndarray, float]]):
        """
        Initialize planning state for a new problem.
        
        Parameters
        ----------
        start : np.ndarray
            Start state [x, y, heading]
        goal : np.ndarray
            Goal state [x, y, heading]
        obstacles : List[Tuple[np.ndarray, float]]
            List of (center, radius) tuples
        """
        # Initialize start node
        self.start_node = AStarNode(
            f=0.0, g=0.0, h=0.0,
            x=start[0], y=start[1], heading=start[2]
        )
        self.start_node.h = self._heuristic(self.start_node, goal)
        self.start_node.f = self.start_node.g + self.start_node.h
        
        # Reset planning state
        self.open_set = [self.start_node]
        self.closed_set = set()
        self.g_score = {self.start_node.grid_pos: 0.0}
        self.goal = goal
        self.obstacles = obstacles
        self.iterations = 0
        self.nodes_expanded = 0
        self.best_node = self.start_node
        self.status = PlannerStatus.PLANNING
        
        # Reset visualization data
        self.closed_nodes = []
        self.open_nodes = [self.start_node]
        self.current_node = None
        
        print(f"A* initialized: {start[:3]} → {goal[:3]}")
        print(f"Grid: {cfg.GRID_RESOLUTION}m, Heading: {np.degrees(cfg.HEADING_RESOLUTION):.1f}°")
    
    def step(self, time_budget: float = None) -> PlannerStatus:
        """
        Execute planning for a time budget or number of iterations.
        
        Parameters
        ----------
        time_budget : float, optional
            Time budget in seconds. If None, uses cfg.TIME_BUDGET_PER_STEP
            
        Returns
        -------
        PlannerStatus
            Current status of the planner
        """
        if self.status != PlannerStatus.PLANNING:
            return self.status
        
        if time_budget is None:
            time_budget = cfg.TIME_BUDGET_PER_STEP
        
        start_time = time.time()
        iterations_this_step = 0
        
        while self.open_set and self.iterations < cfg.MAX_ITERATIONS:
            self.iterations += 1
            iterations_this_step += 1
            
            # Get node with lowest f-cost
            current = heapq.heappop(self.open_set)
            
            # Skip if already expanded
            if current.grid_pos in self.closed_set:
                continue
            
            # Update visualization
            self.current_node = current
            if len(self.closed_nodes) < cfg.MAX_NODES_TO_DRAW:
                self.closed_nodes.append(current)
            
            # Update best node (closest to goal)
            if current.h < self.best_node.h:
                self.best_node = current
            
            # Check if goal reached
            if self._is_goal_reached(current, self.goal):
                self.status = PlannerStatus.PATH_FOUND
                print(f"A* found path: {self.iterations} iterations, {self.nodes_expanded} nodes expanded")
                return self.status
            
            # Mark as expanded
            self.closed_set.add(current.grid_pos)
            self.nodes_expanded += 1
            
            # Check time budget periodically
            if iterations_this_step % cfg.ITERATIONS_PER_CHECK == 0:
                elapsed = time.time() - start_time
                if elapsed >= time_budget:
                    # Don't change status - just return to allow continuation
                    return PlannerStatus.TIME_EXPIRED
            
            # Expand neighbors using motion primitives
            for forward_dist, turn_angle in self.motion_primitives:
                # Apply motion primitive
                new_x, new_y, new_heading = self._apply_motion_primitive(
                    current, forward_dist, turn_angle
                )
                
                # Check bounds
                if not self._in_bounds(new_x, new_y):
                    continue
                
                # Create neighbor node
                neighbor = AStarNode(
                    f=0.0, g=0.0, h=0.0,
                    x=new_x, y=new_y, heading=new_heading,
                    parent=current
                )
                
                # Skip if in closed set
                if neighbor.grid_pos in self.closed_set:
                    continue
                
                # Check collision
                if not self._check_path_collision(current, neighbor, self.obstacles):
                    continue
                
                # Compute costs
                step_dist = abs(forward_dist)
                turn_cost = abs(turn_angle) * cfg.COST_TURN
                reverse_cost = cfg.COST_REVERSE if forward_dist < 0 else 0.0
                
                tentative_g = current.g + cfg.COST_STEP * step_dist + turn_cost + reverse_cost
                
                # Check if this is better path to this grid cell
                if neighbor.grid_pos in self.g_score:
                    if tentative_g >= self.g_score[neighbor.grid_pos]:
                        continue
                
                # Update costs
                neighbor.g = tentative_g
                neighbor.h = self._heuristic(neighbor, self.goal)
                neighbor.f = neighbor.g + neighbor.h
                
                self.g_score[neighbor.grid_pos] = tentative_g
                heapq.heappush(self.open_set, neighbor)
        
        # Update open nodes visualization (sample for performance)
        if len(self.open_set) <= cfg.MAX_NODES_TO_DRAW:
            self.open_nodes = list(self.open_set)
        else:
            # Sample open set for visualization
            step = len(self.open_set) // cfg.MAX_NODES_TO_DRAW
            self.open_nodes = [self.open_set[i] for i in range(0, len(self.open_set), step)]
        
        # Exhausted open set without finding goal
        self.status = PlannerStatus.NO_PATH
        print(f"A* exhausted search after {self.iterations} iterations")
        return self.status
    
    def get_current_path(self) -> Optional[List[np.ndarray]]:
        """
        Get current best path (may be incomplete if planning is ongoing).
        
        Returns
        -------
        Optional[List[np.ndarray]]
            Path to best node found so far, or complete path if goal reached
        """
        if self.status == PlannerStatus.NOT_STARTED or self.best_node is None:
            return None
        
        # Return path to best node found so far
        return self._extract_path(self.best_node)
    
    def get_planning_info(self) -> dict:
        """Get current planning statistics."""
        return {
            'status': self.status,
            'iterations': self.iterations,
            'nodes_expanded': self.nodes_expanded,
            'open_set_size': len(self.open_set),
            'closed_set_size': len(self.closed_set),
            'best_node_distance': self.best_node.h if self.best_node else float('inf')
        }
    
    def plan(self, start: np.ndarray, goal: np.ndarray,
            obstacles: List[Tuple[np.ndarray, float]],
            max_time: float = None) -> Optional[List[np.ndarray]]:
        """
        Plan complete path (blocking call).
        
        Parameters
        ----------
        start : np.ndarray
            Start state [x, y, heading]
        goal : np.ndarray
            Goal state [x, y, heading]
        obstacles : List[Tuple[np.ndarray, float]]
            List of (center, radius) tuples
        max_time : float, optional
            Maximum planning time. If None, uses MAX_ITERATIONS
            
        Returns
        -------
        Optional[List[np.ndarray]]
            Complete path or None if not found
        """
        self.initialize_planning(start, goal, obstacles)
        
        if max_time is None:
            # Plan until completion
            while self.status == PlannerStatus.PLANNING:
                self.step(time_budget=1.0)  # Large time budget
        else:
            # Plan with time limit
            start_time = time.time()
            while self.status == PlannerStatus.PLANNING:
                remaining = max_time - (time.time() - start_time)
                if remaining <= 0:
                    break
                self.step(time_budget=min(cfg.TIME_BUDGET_PER_STEP, remaining))
        
        if self.status == PlannerStatus.PATH_FOUND:
            return self.get_current_path()
        
        return None
    
    def _extract_path(self, goal_node: AStarNode) -> List[np.ndarray]:
        """Extract path by backtracking from goal to start."""
        path = []
        current = goal_node
        
        while current is not None:
            path.append(current.state)
            current = current.parent
        
        path.reverse()
        return path
