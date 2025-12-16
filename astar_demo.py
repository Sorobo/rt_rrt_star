"""
Demo of A* planner with pygame visualization.
"""

import sys
import math
import pygame
import numpy as np
import json

from boat_dynamics import MilliAmpere1Sim
from astar_planner import AStarPlanner
from config import WORLD_BOUNDS, BOAT_WIDTH, BOAT_LENGTH
import astar_config as cfg

# Pygame setup
SCREEN_SIZE = 800
FPS = 60

# Colors
COLOR_BG          = (25, 25, 25)
COLOR_GRID        = (40, 40, 40)
COLOR_PATH        = (255, 215, 0)
COLOR_AGENT       = (50, 150, 255)
COLOR_GOAL        = (50, 255, 100)
COLOR_OBSTACLE    = (200, 60, 60)
COLOR_CLOSED_NODE = (80, 40, 100)   # Purple for explored nodes
COLOR_OPEN_NODE   = (40, 80, 120)   # Blue for open set nodes
COLOR_CURRENT     = (255, 100, 100) # Red for current expanding node

def world_to_screen(x):
    """Map world coords to screen coords."""
    sx = int((x[0] - WORLD_BOUNDS[0, 0]) /
             (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0]) * SCREEN_SIZE)
    sy = int((x[1] - WORLD_BOUNDS[1, 0]) /
             (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0]) * SCREEN_SIZE)
    sy = SCREEN_SIZE - sy
    return sx, sy

def draw_obstacles(screen, obstacles, rectangles=None):
    # Draw circles
    for center, radius in obstacles:
        cx, cy = world_to_screen(center)
        r = int(radius / (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0]) * SCREEN_SIZE)
        pygame.draw.circle(screen, COLOR_OBSTACLE, (cx, cy), r, width=0)
    
    # Draw rectangles
    if rectangles:
        for corner1, corner2 in rectangles:
            x1, y1 = world_to_screen(corner1)
            x2, y2 = world_to_screen(corner2)
            
            left = min(x1, x2)
            top = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            pygame.draw.rect(screen, COLOR_OBSTACLE, (left, top, width, height), width=0)

def draw_boat(screen, boat):
    """Draw the boat as a rectangle with heading indicator."""
    corners = boat.get_corners()
    screen_corners = [world_to_screen(corner) for corner in corners]
    
    boat_surface = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE), pygame.SRCALPHA)
    pygame.draw.polygon(boat_surface, (*COLOR_AGENT, 128), screen_corners)
    pygame.draw.polygon(boat_surface, (255, 255, 255), screen_corners, 2)
    screen.blit(boat_surface, (0, 0))
    
    center = world_to_screen(boat.x[:2])
    front_center = world_to_screen(boat.x[:2] + np.array([
        np.cos(boat.x[2]) * boat.length / 2,
        np.sin(boat.x[2]) * boat.length / 2
    ]))
    pygame.draw.line(screen, (255, 255, 0), center, front_center, 3)

def draw_goal(screen, x_goal):
    gx, gy = world_to_screen(x_goal[:2])
    pygame.draw.circle(screen, COLOR_GOAL, (gx, gy), 8)
    
    if len(x_goal) >= 3:
        heading = x_goal[2]
        arrow_len = 25
        end_x = gx + arrow_len * math.cos(heading)
        end_y = gy - arrow_len * math.sin(heading)
        
        pygame.draw.line(screen, COLOR_GOAL, (gx, gy), (int(end_x), int(end_y)), 3)
        
        arrow_head_len = 10
        arrow_angle = math.pi / 6
        
        p1 = (int(end_x - arrow_head_len * math.cos(heading - arrow_angle)),
              int(end_y + arrow_head_len * math.sin(heading - arrow_angle)))
        p2 = (int(end_x - arrow_head_len * math.cos(heading + arrow_angle)),
              int(end_y + arrow_head_len * math.sin(heading + arrow_angle)))
        
        pygame.draw.line(screen, COLOR_GOAL, (int(end_x), int(end_y)), p1, 3)
        pygame.draw.line(screen, COLOR_GOAL, (int(end_x), int(end_y)), p2, 3)

def draw_path(screen, path):
    """Draw the planned path."""
    if path is None or len(path) < 2:
        return
    
    # Draw path line
    pts = [world_to_screen(state[:2]) for state in path]
    pygame.draw.lines(screen, COLOR_PATH, False, pts, width=3)
    
    # Draw waypoints with heading indicators
    for i, state in enumerate(path):
        if i % 3 == 0:  # Every 3rd waypoint
            pos = world_to_screen(state[:2])
            pygame.draw.circle(screen, COLOR_PATH, pos, 3)
            
            # Heading indicator
            heading = state[2]
            line_len = 12
            end_x = pos[0] + line_len * math.cos(heading)
            end_y = pos[1] - line_len * math.sin(heading)
            pygame.draw.line(screen, COLOR_PATH, pos, (int(end_x), int(end_y)), 2)

def draw_grid(screen):
    """Draw planning grid."""
    grid_size = cfg.GRID_RESOLUTION
    
    for i in np.arange(WORLD_BOUNDS[0, 0], WORLD_BOUNDS[0, 1] + grid_size, grid_size):
        x = (i - WORLD_BOUNDS[0, 0]) / (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0]) * SCREEN_SIZE
        pygame.draw.line(screen, COLOR_GRID, (x, 0), (x, SCREEN_SIZE), 1)
    
    for i in np.arange(WORLD_BOUNDS[1, 0], WORLD_BOUNDS[1, 1] + grid_size, grid_size):
        y = (i - WORLD_BOUNDS[1, 0]) / (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0]) * SCREEN_SIZE
        pygame.draw.line(screen, COLOR_GRID, (0, SCREEN_SIZE - y), (SCREEN_SIZE, SCREEN_SIZE - y), 1)

def draw_planning_nodes(screen, planner):
    """Draw explored and open nodes from the planner."""
    if not cfg.SHOW_CLOSED_NODES and not cfg.SHOW_OPEN_NODES:
        return
    
    # Draw closed nodes (explored) - sample if too many
    if cfg.SHOW_CLOSED_NODES and planner.closed_nodes:
        nodes_to_draw = planner.closed_nodes
        if len(nodes_to_draw) > cfg.MAX_NODES_TO_DRAW:
            # Sample evenly across the list
            step = len(nodes_to_draw) // cfg.MAX_NODES_TO_DRAW
            nodes_to_draw = nodes_to_draw[::step]
        
        for node in nodes_to_draw:
            pos = world_to_screen(np.array([node.x, node.y]))
            pygame.draw.circle(screen, COLOR_CLOSED_NODE, pos, 2)
    
    # Draw open nodes (frontier) - sample if too many
    if cfg.SHOW_OPEN_NODES and planner.open_nodes:
        nodes_to_draw = planner.open_nodes
        if len(nodes_to_draw) > 100:  # Limit open nodes even more
            step = len(nodes_to_draw) // 100
            nodes_to_draw = nodes_to_draw[::step]
        
        for node in nodes_to_draw:
            pos = world_to_screen(np.array([node.x, node.y]))
            pygame.draw.circle(screen, COLOR_OPEN_NODE, pos, 3)
    
    # Draw current node being expanded
    if cfg.SHOW_CURRENT_NODE and planner.current_node is not None:
        pos = world_to_screen(np.array([planner.current_node.x, planner.current_node.y]))
        pygame.draw.circle(screen, COLOR_CURRENT, pos, 5)
        
        # Draw heading indicator for current node
        heading = planner.current_node.heading
        line_len = 15
        end_x = pos[0] + line_len * math.cos(heading)
        end_y = pos[1] - line_len * math.sin(heading)
        pygame.draw.line(screen, COLOR_CURRENT, pos, (int(end_x), int(end_y)), 2)

def load_map_obstacles(filename="map_obstacles.json"):
    """Load obstacles from JSON file."""
    circles = []
    rectangles = []
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Add circles
        for circle in data.get('circles', []):
            center = np.array(circle['center'])
            radius = circle['radius']
            circles.append((center, radius))
        
        # Add rectangles
        for rect in data.get('rectangles', []):
            corner1 = np.array(rect['corner1'])
            corner2 = np.array(rect['corner2'])
            rectangles.append((corner1, corner2))
        
        print(f"Loaded {len(circles)} circles and {len(rectangles)} rectangles from {filename}")
    except FileNotFoundError:
        print(f"Map file {filename} not found, using empty map")
    except Exception as e:
        print(f"Error loading map: {e}")
    
    return circles, rectangles

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("A* Path Planner")
    clock = pygame.time.Clock()
    
    # Setup
    x_start = np.array([5.0, 5.0, 0.0])
    x_goal = np.array([25.0, 25.0, np.pi/4])
    
    obstacles, rectangles = load_map_obstacles()
    
    boat = MilliAmpere1Sim(np.append(x_start, [0.0, 0.0, 0.0]))
    planner = AStarPlanner()
    
    # Display state
    show_grid = True
    show_planning_info = True
    
    # Initialize planning (real-time mode)
    print("Starting real-time A* planning...")
    planner.initialize_planning(x_start, x_goal, obstacles, rectangles)
    path = None
    
    # Animation state
    path_index = 0
    boat_moving = True  # Auto-move along path
    planning_active = True
    
    # Goal setting state
    dragging_goal = False
    goal_start_pos = None
    
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Keyboard controls
            if event.type == pygame.KEYDOWN:
                # R to replan
                if event.key == pygame.K_r:
                    print("\nReplanning...")
                    planner.initialize_planning(boat.x[:3], x_goal, obstacles, rectangles)
                    planning_active = True
                    path_index = 0
                
                # P to pause/resume planning
                if event.key == pygame.K_p:
                    planning_active = not planning_active
                    status = "resumed" if planning_active else "paused"
                    print(f"Planning {status}")
                
                # G to toggle grid display
                if event.key == pygame.K_g:
                    show_grid = not show_grid
                
                # I to toggle planning info
                if event.key == pygame.K_i:
                    show_planning_info = not show_planning_info
                
                # V to toggle visualization of nodes
                if event.key == pygame.K_v:
                    cfg.SHOW_CLOSED_NODES = not cfg.SHOW_CLOSED_NODES
                    cfg.SHOW_OPEN_NODES = not cfg.SHOW_OPEN_NODES
                    status = "ON" if cfg.SHOW_CLOSED_NODES else "OFF"
                    print(f"Node visualization: {status}")
            
            # Left-click and drag to set goal
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                wx = mx / SCREEN_SIZE * (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0])
                wy = (SCREEN_SIZE - my) / SCREEN_SIZE * (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0])
                goal_start_pos = np.array([wx, wy])
                dragging_goal = True
            
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if dragging_goal and goal_start_pos is not None:
                    mx, my = pygame.mouse.get_pos()
                    wx = mx / SCREEN_SIZE * (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0])
                    wy = (SCREEN_SIZE - my) / SCREEN_SIZE * (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0])
                    goal_end_pos = np.array([wx, wy])
                    
                    drag_vector = goal_end_pos - goal_start_pos
                    if np.linalg.norm(drag_vector) > 0.5:
                        heading = np.arctan2(drag_vector[1], drag_vector[0])
                    else:
                        heading = 0.0
                    
                    x_goal = np.array([goal_start_pos[0], goal_start_pos[1], heading])
                    print(f"\nNew goal: ({x_goal[0]:.1f}, {x_goal[1]:.1f}, {np.degrees(heading):.1f}°)")
                    
                    # Replan
                    planner.initialize_planning(boat.x[:3], x_goal, obstacles, rectangles)
                    planning_active = True
                    path_index = 0
                
                dragging_goal = False
                goal_start_pos = None
            
            # Right-click to add obstacle
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                mx, my = event.pos
                wx = mx / SCREEN_SIZE * (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0])
                wy = (SCREEN_SIZE - my) / SCREEN_SIZE * (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0])
                obstacles.append((np.array([wx, wy]), 1.5))
                print(f"Added obstacle at ({wx:.1f}, {wy:.1f})")
        
        # Real-time planning step
        from astar_planner import PlannerStatus
        if planning_active and (planner.status == PlannerStatus.PLANNING or 
                                planner.status == PlannerStatus.TIME_EXPIRED):
            status = planner.step()
            
            # Update path with intermediate result
            path = planner.get_current_path()
            
            if status == PlannerStatus.PATH_FOUND:
                print(f"Complete path found!")
                info = planner.get_planning_info()
                print(f"  Iterations: {info['iterations']}, Nodes: {info['nodes_expanded']}")
                planning_active = False
            elif status == PlannerStatus.NO_PATH:
                print("No path exists!")
                planning_active = False
            # TIME_EXPIRED is expected - just continue planning next frame
        
        # Animate boat along path at slow speed (framerate-independent)
        if boat_moving and path is not None and path_index < len(path):
            target_state = path[path_index]
            
            # Move towards target at slow speed
            dist = np.linalg.norm(boat.x[:2] - target_state[:2])
            if dist < 0.5:  # Reached waypoint
                path_index += 1
                if path_index >= len(path):
                    print("Goal reached!")
            else:
                # Interpolate slowly towards target (framerate-independent)
                # Base speed: 1.0 means full interpolation in 1 second at 60 FPS
                alpha = 1.0 * dt  # Scale with delta time
                alpha = min(alpha, 1.0)  # Clamp to prevent overshoot
                boat.x[0] = boat.x[0] + alpha * (target_state[0] - boat.x[0])
                boat.x[1] = boat.x[1] + alpha * (target_state[1] - boat.x[1])
                # Interpolate heading to avoid discontinuities
                angle_diff = target_state[2] - boat.x[2]
                # Normalize to [-pi, pi]
                angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
                boat.x[2] = boat.x[2] + alpha * angle_diff
        
        # Drawing
        screen.fill(COLOR_BG)
        
        if show_grid:
            draw_grid(screen)
        draw_obstacles(screen, obstacles, rectangles)
        
        # Draw planning visualization
        draw_planning_nodes(screen, planner)
        
        draw_path(screen, path)
        draw_boat(screen, boat)
        draw_goal(screen, x_goal)
        
        # Draw goal direction preview
        if dragging_goal and goal_start_pos is not None:
            mx, my = pygame.mouse.get_pos()
            wx = mx / SCREEN_SIZE * (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0])
            wy = (SCREEN_SIZE - my) / SCREEN_SIZE * (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0])
            
            start_screen = world_to_screen(goal_start_pos)
            end_screen = world_to_screen(np.array([wx, wy]))
            
            pygame.draw.line(screen, (255, 255, 0), start_screen, end_screen, 3)
            pygame.draw.circle(screen, (255, 255, 0), start_screen, 8, 2)
        
        # Instructions
        font = pygame.font.Font(None, 24)
        instructions = [
            "R: Replan | P: Pause planning",
            "V: Nodes | G: Grid | I: Info",
            "Left-drag: Goal | Right-click: Obstacle"
        ]
        for i, text in enumerate(instructions):
            surf = font.render(text, True, (200, 200, 200))
            screen.blit(surf, (10, 10 + i * 25))
        
        # Show planning info
        if show_planning_info:
            info = planner.get_planning_info()
            from astar_planner import PlannerStatus
            
            status_text = f"Status: {info['status'].value}"
            iter_text = f"Iterations: {info['iterations']}"
            nodes_text = f"Nodes: {info['nodes_expanded']}"
            open_text = f"Open: {info['open_set_size']}"
            dist_text = f"Best dist: {info['best_node_distance']:.2f}m"
            
            y_offset = SCREEN_SIZE - 150
            for i, text in enumerate([status_text, iter_text, nodes_text, open_text, dist_text]):
                surf = font.render(text, True, (200, 200, 200))
                screen.blit(surf, (10, y_offset + i * 25))
        
        # Show path info
        if path is not None:
            info_text = f"Path: {len(path)} waypoints"
            surf = font.render(info_text, True, (255, 215, 0))
            screen.blit(surf, (10, SCREEN_SIZE - 30))
        
        # Show legend for node colors (right side)
        if cfg.SHOW_CLOSED_NODES or cfg.SHOW_OPEN_NODES:
            legend_x = SCREEN_SIZE - 180
            legend_y = 10
            font_small = pygame.font.Font(None, 20)
            
            if cfg.SHOW_CLOSED_NODES:
                pygame.draw.circle(screen, COLOR_CLOSED_NODE, (legend_x, legend_y), 4)
                text = font_small.render("Explored", True, (200, 200, 200))
                screen.blit(text, (legend_x + 10, legend_y - 8))
                legend_y += 20
            
            if cfg.SHOW_OPEN_NODES:
                pygame.draw.circle(screen, COLOR_OPEN_NODE, (legend_x, legend_y), 4)
                text = font_small.render("Frontier", True, (200, 200, 200))
                screen.blit(text, (legend_x + 10, legend_y - 8))
                legend_y += 20
            
            if cfg.SHOW_CURRENT_NODE:
                pygame.draw.circle(screen, COLOR_CURRENT, (legend_x, legend_y), 4)
                text = font_small.render("Current", True, (200, 200, 200))
                screen.blit(text, (legend_x + 10, legend_y - 8))
        
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
