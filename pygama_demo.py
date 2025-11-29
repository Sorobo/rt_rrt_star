import sys
import math
import pygame
import numpy as np

from agent_dynamics import AgentDynamics
from rt_rrt_star import RTRRTStar
from config import WORLD_BOUNDS, OBSTACLE_BLOCK_RADIUS,R_S
from dynamic_obstacle import DynamicObstacle

# ---------------------------
# Pygame / visualization setup
# ---------------------------

SCREEN_SIZE = 800
FPS = 60



# Colors
COLOR_BG       = (25, 25, 25)
COLOR_TREE     = (80, 80, 80)
COLOR_PATH     = (255, 215, 0)
COLOR_AGENT    = (50, 150, 255)
COLOR_GOAL     = (50, 255, 100)
COLOR_OBSTACLE = (200, 60, 60)
COLOR_NODE     = (150, 150, 150)


def world_to_screen(x):
    """Map world coords [0,100]x[0,100] to screen [0,800]x[0,800]."""
    sx = int((x[0] - WORLD_BOUNDS[0, 0]) /
             (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0]) * SCREEN_SIZE)
    sy = int((x[1] - WORLD_BOUNDS[1, 0]) /
             (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0]) * SCREEN_SIZE)
    # Pygame y-axis is down, so flip:
    sy = SCREEN_SIZE - sy
    return sx, sy


def draw_obstacles(screen, obstacles):
    for center, radius in obstacles:
        cx, cy = world_to_screen(center)
        # scale radius
        r = int(radius / (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0]) * SCREEN_SIZE)
        pygame.draw.circle(screen, COLOR_OBSTACLE, (cx, cy), r, width=0)

def draw_dynamic_obstacles(screen, dynamic_obstacles):
    """Draw dynamic obstacles with their blocking radius"""
    for dyn_obs in dynamic_obstacles:
        cx, cy = world_to_screen(dyn_obs.center)
        
        # Scale radii
        r = int(dyn_obs.radius / (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0]) * SCREEN_SIZE)
        block_r = int((dyn_obs.radius + OBSTACLE_BLOCK_RADIUS) / 
                     (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0]) * SCREEN_SIZE)
        
        # Draw outer blocking radius (semi-transparent)
        s = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(s, (255, 100, 100, 50), (cx, cy), block_r)
        screen.blit(s, (0, 0))
        
        # Draw obstacle itself
        pygame.draw.circle(screen, (255, 50, 50), (cx, cy), r, width=0)
        
        # Draw velocity arrow
        vel_scale = 10  # Scale factor for visibility
        end_x = int(cx + dyn_obs.velocity[0] * vel_scale)
        end_y = int(cy - dyn_obs.velocity[1] * vel_scale)  # Flip y
        pygame.draw.line(screen, (255, 255, 0), (cx, cy), (end_x, end_y), 2)


def draw_tree(screen, tree):
    
    # Draw edges first
    for node_id in tree.index.nodes:  # Iterate over node IDs
        node = tree.index.nodes[node_id]  # Fetch the actual Node object
        if node.parent is not None:
            #if node.blocked or node.parent.blocked:
            #    continue  # Skip drawing edges for blocked nodes or to blocked parents
            x1 = world_to_screen(node.parent.x)
            x2 = world_to_screen(node.x)
            if node.blocked or node.parent.blocked:
                pygame.draw.line(screen, (255, 0, 255), x1, x2, width=1)  # Magenta for blocked
            else:
                pygame.draw.line(screen, COLOR_TREE, x1, x2, width=1)
            
            # Draw arrowhead
            angle = math.atan2(x2[1] - x1[1], x2[0] - x1[0])
            arrow_len = 3
            arrow_angle = math.pi / 6
            
            p1 = (x2[0] - arrow_len * math.cos(angle - arrow_angle),
                  x2[1] - arrow_len * math.sin(angle - arrow_angle))
            p2 = (x2[0] - arrow_len * math.cos(angle + arrow_angle),
                  x2[1] - arrow_len * math.sin(angle + arrow_angle))
            
            pygame.draw.line(screen, COLOR_TREE, x2, p1, width=1)
            pygame.draw.line(screen, COLOR_TREE, x2, p2, width=1)


    # Draw nodes
    for node_id in tree.index.nodes:  # Iterate over node IDs
        node = tree.index.nodes[node_id]  # Fetch the actual Node object
        #if node.blocked:
        #    continue  # Skip drawing blocked nodes
        x = world_to_screen(node.x)
        #if node.blocked:
        #    pygame.draw.circle(screen, (255, 0, 255), x, 2)  # Magenta for blocked
        if node.cost == 0:
                pygame.draw.circle(screen, (255, 0, 0), x, 4)
        else:
            # Color based on cost
            if node.cost == float("inf"):
                pygame.draw.circle(screen, (255, 165, 0), x, 2)
            else:
                pygame.draw.circle(screen, COLOR_TREE, x, 2)
    # Draw nodes
    """
    for node_id in tree.index.nodes:  # Iterate over node IDs
        node = tree.index.nodes[node_id]
        x = world_to_screen(node.x)

        pygame.draw.circle(screen, COLOR_NODE, x, 2)
        #pygame.draw.circle(screen, COLOR_NODE, x, 2)
    """

def draw_path(screen, path):
    if len(path) < 2:
        return
    pts = [world_to_screen(n.x) for n in path]
    pygame.draw.lines(screen, COLOR_PATH, False, pts, width=3)


def draw_agent_and_goal(screen, x_agent, x_goal):
    ax, ay = world_to_screen(x_agent)
    gx, gy = world_to_screen(x_goal)

    pygame.draw.circle(screen, COLOR_AGENT, (ax, ay), 6)
    pygame.draw.circle(screen, COLOR_GOAL, (gx, gy), 6)

def draw_ruler(screen):
    # Draw vertical grid lines
    for i in range(0, 101, 1):  # Assuming world bounds are [0, 100] x [0, 100]
        x = i / (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0]) * SCREEN_SIZE
        pygame.draw.line(screen, (50, 50, 50), (x, 0), (x, SCREEN_SIZE), 1)

    # Draw horizontal grid lines
    for i in range(0, 101, 1):
        y = i / (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0]) * SCREEN_SIZE
        pygame.draw.line(screen, (50, 50, 50), (0, SCREEN_SIZE - y), (SCREEN_SIZE, SCREEN_SIZE - y), 1)

def draw_root_node(screen, tree):
    if tree.root is not None:
        root = tree.root
        x = world_to_screen(root.x)
        pygame.draw.circle(screen, (128, 0, 128), x, 5)  # Bright purple

def draw_sampling_ellipse(screen, c_best, x_start, x_goal):
    if c_best == float("inf"):
        return  # No valid path yet

    c_min = np.linalg.norm(x_start - x_goal)
    if c_best < c_min:
        return  # Invalid case

    # Ellipse major/minor axes
    a = c_best / 2.0
    b = np.sqrt(c_best**2 - c_min**2) / 2.0

    # Ellipse center
    center = (x_start + x_goal) / 2.0

    # Angle of ellipse
    angle = math.atan2(x_goal[1] - x_start[1], x_goal[0] - x_start[0])

    # Draw ellipse
    num_points = 100
    points = []
    for t in np.linspace(0, 2 * np.pi, num_points):
        x_ellipse = a * math.cos(t)
        y_ellipse = b * math.sin(t)

        # Rotate and translate
        x_rotated = (x_ellipse * math.cos(angle) - y_ellipse * math.sin(angle)) + center[0]
        y_rotated = (x_ellipse * math.sin(angle) + y_ellipse * math.cos(angle)) + center[1]

        points.append(world_to_screen(np.array([x_rotated, y_rotated])))

    pygame.draw.lines(screen, (0, 255, 255), True, points, 1)

def draw_search_radius(screen, tree):
    if tree.root is None:
        return
    center = world_to_screen(tree.root.x)
    radius = R_S / (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0]) * SCREEN_SIZE
    pygame.draw.circle(screen, (0, 255, 0), center, int(radius), 1)


def generate_obstacles(num_obstacles, min_radius, max_radius):
    obstacles = []
    for _ in range(num_obstacles):
        x = np.random.uniform(WORLD_BOUNDS[0, 0], WORLD_BOUNDS[0, 1])
        y = np.random.uniform(WORLD_BOUNDS[1, 0], WORLD_BOUNDS[1, 1])
        radius = np.random.uniform(min_radius, max_radius)
        obstacles.append((np.array([x, y]), radius))
    return obstacles
def main():
    
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("RT-RRT* Visualization")
    clock = pygame.time.Clock()

    # ------------------------------------
    # Define start, goal, and obstacles
    # ------------------------------------
    x_start = np.array([0.5, 0.5])
    x_goal  = np.array([0.8, 0.8])

    obstacles = [
        (np.array([0.0, 5.0]), 3.0),
        (np.array([10.0, 10.0]), 1.0),
        (np.array([7.0, 4.0]), 2.0),
        (np.array([3.0, 3.0]), 1.0),
        (np.array([7.0, 7.0]), 1.5),
        (np.array([12.0, 6.0]), 1.0),
        (np.array([6.0, 12.0]), 1.2),
        (np.array([9.0, 3.0]), 0.8),
        (np.array([4.0, 8.0]), 1.0),
        (np.array([11.0, 13.0]), 1.5),
        (np.array([14.0, 5.0]), 1.3),
        (np.array([2.0, 11.0]), 0.9),
        (np.array([8.0, 14.0]), 1.1),
        (np.array([13.0, 9.0]), 1.2),
        (np.array([5.0, 2.0]), 0.7),
        (np.array([10.0, 1.0]), 1.0),
        (np.array([15.0, 15.0]), 1.5),
        (np.array([10.0, 5.0]), 2.0),
        (np.array([5.0, 10.0]), 1.0),
        (np.array([10.0, 10.0]), 1.2),
        (np.array([5.0, 5.0]), 1.0),
        (np.array([10.0, 15.0]), 1.5),
        (np.array([15.0, 10.0]), 1.3),
        (np.array([10.0, 5.0]), 1.1),
        (np.array([5.0, 10.0]), 0.9),
        (np.array([10.0, 10.0]), 1.4),
        (np.array([4, 6]), 1.0)
    ]

    obstacles = generate_obstacles(25, 1, 2)
    # Initialize dynamic obstacles
    dynamic_obstacles = [
        #DynamicObstacle(center=[5.0, 7.0], radius=0.8, velocity=[.15, .08]),
        #DynamicObstacle(center=[10.0, 3.0], radius=0.6, velocity=[-0.10, .12]),
        #DynamicObstacle(center=[12.0, 12.0], radius=0.7, velocity=[.05, -.15]),
    ]

    planner = RTRRTStar(WORLD_BOUNDS, x_start)
    agent = AgentDynamics(x_start, speed=4)
    x_agent = agent.x
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0  # seconds per frame

        # ---------------
        # Event handling
        # ---------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Optional: left-click to move goal
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # Convert screen to world
                wx = mx / SCREEN_SIZE * (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0])
                wy = (SCREEN_SIZE - my) / SCREEN_SIZE * (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0])
                x_goal = np.array([wx, wy], dtype=float)
                print(f"New goal: {x_goal}")
            # Right-click to add obstacle
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                mx, my = event.pos
                # Convert screen to world
                wx = mx / SCREEN_SIZE * (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0])
                wy = (SCREEN_SIZE - my) / SCREEN_SIZE * (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0])
                obstacles.append((np.array([wx, wy], dtype=float), 1.0))
                print(f"Added obstacle at: ({wx:.2f}, {wy:.2f})")
        # ---------------------------
        # RT-RRT* planning step
        # ---------------------------
        path = planner.step(x_agent, x_goal, obstacles, dynamic_obstacles, dt)

        # Move agent along path (if there is at least 1 step ahead)
        
        if len(path) > 1:
            next_node = path[0]
            x_agent = agent.update(next_node.x, dt)

        # Re-root the tree every second
        # ---------------------------
        # Drawing
        # ---------------------------
        screen.fill(COLOR_BG)

        # Draw environment
        draw_obstacles(screen, obstacles)
        draw_dynamic_obstacles(screen, dynamic_obstacles)
        draw_tree(screen, planner.tree)
        draw_path(screen, path)
        draw_agent_and_goal(screen, x_agent, x_goal)
        #draw_ruler(screen)
        draw_root_node(screen, planner.tree)
        draw_sampling_ellipse(screen, planner.c_best, planner.tree.root.x, x_goal)
        draw_search_radius(screen, planner.tree)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
