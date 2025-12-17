import sys
import math
import pygame
import numpy as np
import json

from agent_dynamics import AgentDynamics
from boat_dynamics import MilliAmpere1Sim
from rt_rrt_star import RTRRTStar
from config import WORLD_BOUNDS, OBSTACLE_BLOCK_RADIUS, R_S, BOAT_WIDTH, BOAT_LENGTH, BOAT_SAFETY_PADDING
from dynamic_obstacle import DynamicObstacle

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


def draw_obstacles(screen, obstacles, rectangles=None):
    # Draw circles
    for center, radius in obstacles:
        cx, cy = world_to_screen(center)
        # scale radius
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
            x1 = world_to_screen(node.parent.x)
            x2 = world_to_screen(node.x)
            # Draw blocked edges (infinite cost) in magenta
            if node.cost == float("inf") or node.parent.cost == float("inf"):
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


    # First pass: collect costs for color mapping
    costs = []
    for node_id in tree.index.nodes:
        node = tree.index.nodes[node_id]
        if node.cost != float("inf") and node.cost > 0:
            costs.append(node.cost)
    
    # Determine cost range for color mapping
    if costs:
        min_cost = min(costs)
        max_cost = max(costs)
        cost_range = max_cost - min_cost if max_cost > min_cost else 1.0
    else:
        min_cost = 0
        cost_range = 1.0
    
    # Draw nodes with cost-based coloring
    for node_id in tree.index.nodes:  # Iterate over node IDs
        node = tree.index.nodes[node_id]  # Fetch the actual Node object
        x = world_to_screen(node.x)
        
        # Determine color based on cost
        if node.cost == 0:
            # Root node - red
            color = (255, 0, 0)
            radius = 4
        elif node.cost == float("inf"):
            # Infinite cost (blocked) - orange
            color = (255, 165, 0)
            radius = 2
        else:
            # Normal nodes - color gradient from green (low cost) to blue (high cost)
            normalized_cost = (node.cost - min_cost) / cost_range
            # Green to cyan to blue gradient
            r = int(50 * (1 - normalized_cost))
            g = int(150 + 105 * (1 - normalized_cost))
            b = int(100 + 155 * normalized_cost)
            color = (r, g, b)
            radius = 2
        
        pygame.draw.circle(screen, color, x, radius)
        
        # Draw heading indicator for nodes with heading info
        if node.cost != float("inf") and len(node.x) >= 3:
            heading = node.x[2]
            line_len = 10  # Length in screen pixels
            end_x = x[0] + line_len * math.cos(heading)
            end_y = x[1] - line_len * math.sin(heading)  # Flip y for pygame
            pygame.draw.line(screen, color, x, (int(end_x), int(end_y)), 1)
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

def draw_collision_hulls(screen, path):
    """Draw convex hulls used for collision detection between consecutive path nodes"""
    from collision import get_convex_hull_between_poses
    from config import BOAT_WIDTH, BOAT_LENGTH
    
    if len(path) < 2:
        return
    
    for i in range(len(path) - 1):
        start_node = path[i]
        end_node = path[i + 1]
        
        # Get start and end positions and headings
        start_pos = start_node.x[:2]
        start_heading = start_node.x[2] if len(start_node.x) > 2 else 0
        end_pos = end_node.x[:2]
        end_heading = end_node.x[2] if len(end_node.x) > 2 else 0
        
        # Get convex hull vertices
        hull_vertices = get_convex_hull_between_poses(
            start_pos, start_heading,
            end_pos, end_heading,
            BOAT_WIDTH, BOAT_LENGTH
        )
        
        # Convert to screen coordinates
        screen_vertices = [world_to_screen(v) for v in hull_vertices]
        
        # Draw the convex hull as a semi-transparent polygon
        if len(screen_vertices) >= 3:
            hull_surface = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE), pygame.SRCALPHA)
            pygame.draw.polygon(hull_surface, (100, 200, 255, 80), screen_vertices)
            screen.blit(hull_surface, (0, 0))
            # Draw outline
            pygame.draw.polygon(screen, (50, 150, 255), screen_vertices, 2)


def draw_collision_hulls(screen, path):
    """Draw convex hulls used for collision detection between consecutive path nodes"""
    from collision import get_convex_hull_between_poses
    from config import BOAT_WIDTH, BOAT_LENGTH
    
    if len(path) < 2:
        return
    
    for i in range(len(path) - 1):
        start_node = path[i]
        end_node = path[i + 1]
        
        # Get start and end positions and headings
        start_pos = start_node.x[:2]
        start_heading = start_node.x[2] if len(start_node.x) > 2 else 0
        end_pos = end_node.x[:2]
        end_heading = end_node.x[2] if len(end_node.x) > 2 else 0
        
        # Get convex hull vertices
        hull_vertices = get_convex_hull_between_poses(
            start_pos, start_heading,
            end_pos, end_heading,
            BOAT_WIDTH, BOAT_LENGTH
        )
        
        # Convert to screen coordinates
        screen_vertices = [world_to_screen(v) for v in hull_vertices]
        
        # Draw the convex hull as a semi-transparent polygon
        if len(screen_vertices) >= 3:
            hull_surface = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE), pygame.SRCALPHA)
            pygame.draw.polygon(hull_surface, (100, 200, 255, 80), screen_vertices)
            screen.blit(hull_surface, (0, 0))
            # Draw outline
            pygame.draw.polygon(screen, (50, 150, 255), screen_vertices, 2)

def draw_collision_hulls(screen, path):
    """Draw convex hulls used for collision detection between consecutive path nodes"""
    from collision import create_swept_hull
    
    if len(path) < 2:
        return
    
    for i in range(len(path) - 1):
        start_node = path[i]
        end_node = path[i + 1]
        
        # Get start and end positions and headings
        start_pos = start_node.x[:2]
        start_heading = start_node.x[2] if len(start_node.x) > 2 else 0
        end_pos = end_node.x[:2]
        end_heading = end_node.x[2] if len(end_node.x) > 2 else 0
        
        # Get convex hull vertices with safety padding (same as collision detection)
        safe_width = BOAT_WIDTH + 2 * BOAT_SAFETY_PADDING
        safe_length = BOAT_LENGTH + 2 * BOAT_SAFETY_PADDING
        
        hull_vertices = create_swept_hull(
            start_pos, start_heading,
            end_pos, end_heading,
            safe_width, safe_length
        )
        
        # Convert to screen coordinates
        screen_vertices = [world_to_screen(v) for v in hull_vertices]
        
        # Draw the convex hull as a semi-transparent polygon
        if len(screen_vertices) >= 3:
            hull_surface = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE), pygame.SRCALPHA)
            pygame.draw.polygon(hull_surface, (100, 200, 255, 80), screen_vertices)
            screen.blit(hull_surface, (0, 0))
            # Draw outline
            pygame.draw.polygon(screen, (50, 150, 255), screen_vertices, 2)

def export_trajectory_and_hulls(path, filename="trajectory_export.json"):
    """Export trajectory points and convex hulls to JSON"""
    from collision import create_swept_hull
    import json
    
    export_data = {
        "trajectory": [],
        "convex_hulls": []
    }
    
    # Export trajectory points
    for node in path:
        export_data["trajectory"].append({
            "x": float(node.x[0]),
            "y": float(node.x[1]),
            "heading": float(node.x[2]) if len(node.x) > 2 else 0.0
        })
    
    # Export convex hulls
    if len(path) >= 2:
        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            
            start_pos = start_node.x[:2]
            start_heading = start_node.x[2] if len(start_node.x) > 2 else 0
            end_pos = end_node.x[:2]
            end_heading = end_node.x[2] if len(end_node.x) > 2 else 0
            
            # Get convex hull with padding
            safe_width = BOAT_WIDTH + 2 * BOAT_SAFETY_PADDING
            safe_length = BOAT_LENGTH + 2 * BOAT_SAFETY_PADDING
            
            hull_vertices = create_swept_hull(
                start_pos, start_heading,
                end_pos, end_heading,
                safe_width, safe_length
            )
            
            export_data["convex_hulls"].append({
                "segment": i,
                "vertices": [[float(v[0]), float(v[1])] for v in hull_vertices]
            })
    
    # Write to file
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Exported trajectory and convex hulls to {filename}")

def draw_boat(screen, boat):
    """Draw the boat as a rectangle with heading indicator"""
    from collision import get_boat_corners
    
    # Draw safety padding (inflated boat)
    safe_corners = get_boat_corners(
        boat.x[:2], boat.x[2], 
        boat.width + 2 * BOAT_SAFETY_PADDING, 
        boat.length + 2 * BOAT_SAFETY_PADDING
    )
    safe_screen_corners = [world_to_screen(corner) for corner in safe_corners]
    
    # Draw actual boat
    corners = boat.get_corners()
    screen_corners = [world_to_screen(corner) for corner in corners]
    
    # Draw boat body with safety padding visualization
    boat_surface = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE), pygame.SRCALPHA)
    pygame.draw.polygon(boat_surface, (255, 200, 0, 40), safe_screen_corners)  # Yellow padding
    pygame.draw.polygon(boat_surface, (*COLOR_AGENT, 128), screen_corners)  # Semi-transparent boat
    pygame.draw.polygon(boat_surface, (255, 255, 255), screen_corners, 2)  # White outline
    screen.blit(boat_surface, (0, 0))
    # Draw heading indicator (arrow from center to front)
    center = world_to_screen(boat.x)
    front_center = world_to_screen(boat.x[:2] + np.array([
        np.cos(boat.x[2]) * boat.length / 2,
        np.sin(boat.x[2]) * boat.length / 2
    ]))
    pygame.draw.line(screen, (255, 255, 0), center, front_center, 3)

def draw_agent_and_goal(screen, x_agent, x_goal):
    ax, ay = world_to_screen(x_agent)
    gx, gy = world_to_screen(x_goal)

    pygame.draw.circle(screen, COLOR_AGENT, (ax, ay), 6)
    pygame.draw.circle(screen, COLOR_GOAL, (gx, gy), 6)
    
def draw_goal(screen, x_goal):
    gx, gy = world_to_screen(x_goal)
    pygame.draw.circle(screen, COLOR_GOAL, (gx, gy), 6)
    
    # Draw heading arrow if goal has heading information
    if len(x_goal) >= 3:
        heading = x_goal[2]
        arrow_len = 20  # Length in screen pixels
        end_x = gx + arrow_len * math.cos(heading)
        end_y = gy - arrow_len * math.sin(heading)  # Flip y for pygame
        
        # Draw main arrow line
        pygame.draw.line(screen, COLOR_GOAL, (gx, gy), (int(end_x), int(end_y)), 3)
        
        # Draw arrowhead
        arrow_head_len = 8
        arrow_angle = math.pi / 6
        
        p1 = (int(end_x - arrow_head_len * math.cos(heading - arrow_angle)),
              int(end_y + arrow_head_len * math.sin(heading - arrow_angle)))
        p2 = (int(end_x - arrow_head_len * math.cos(heading + arrow_angle)),
              int(end_y + arrow_head_len * math.sin(heading + arrow_angle)))
        
        pygame.draw.line(screen, COLOR_GOAL, (int(end_x), int(end_y)), p1, 3)
        pygame.draw.line(screen, COLOR_GOAL, (int(end_x), int(end_y)), p2, 3)

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
        # Skip obstacles in bottom left corner (x < 7 and y < 7)
        if x < 13 and y < 7:
            continue
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
    x_start = np.array([4, 10,np.pi/2])
    x_goal  = np.array([0.8, 0.8,0])

    obstacles, rectangles = load_map_obstacles()
    #obstacles, rectangles = [], []
    # Initialize dynamic obstacles
    dynamic_obstacles = [
        #DynamicObstacle(center=[15.0, 0.0], radius=2.5, velocity=[0,0]),
        #DynamicObstacle(center=[15.0, 5.0], radius=2.5, velocity=[-0, 0]),
        #DynamicObstacle(center=[15.0, 10.0], radius=2.5, velocity=[0, 0]),
    ]

    planner = RTRRTStar(WORLD_BOUNDS, x_start)
    boat = MilliAmpere1Sim(x_start)
    x_agent = boat.x
    print("Starting RT-RRT* demo. Close window to exit.",x_agent)
    running = True
    
    # Display state
    show_tree = True
    show_info = True
    show_collision_hulls = False
    
    # Goal setting state
    dragging_goal = False
    goal_start_pos = None
    
    # Dynamic obstacle creation state
    dragging_obstacle = False
    obstacle_start_pos = None

    while running:
        dt = clock.tick(FPS) / 1000.0  # seconds per frame

        # ---------------
        # Event handling
        # ---------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Keyboard controls
            if event.type == pygame.KEYDOWN:
                # T to toggle tree visualization
                if event.key == pygame.K_t:
                    show_tree = not show_tree
                    status = "ON" if show_tree else "OFF"
                    print(f"Tree visualization: {status}")
                
                # I to toggle info display
                if event.key == pygame.K_i:
                    show_info = not show_info
                
                # C to toggle collision hull visualization
                if event.key == pygame.K_c:
                    show_collision_hulls = not show_collision_hulls
                    status = "ON" if show_collision_hulls else "OFF"
                    print(f"Collision hull visualization: {status}")
                
                # D to delete all dynamic obstacles
                if event.key == pygame.K_d:
                    num_obstacles = len(dynamic_obstacles)
                    dynamic_obstacles.clear()
                    print(f"Deleted {num_obstacles} dynamic obstacles")
                
                # E to export trajectory and convex hulls
                if event.key == pygame.K_e:
                    if path:
                        export_trajectory_and_hulls(path)
                    else:
                        print("No path to export")

            # Left-click and drag to set goal position and direction
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # Convert screen to world
                wx = mx / SCREEN_SIZE * (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0])
                wy = (SCREEN_SIZE - my) / SCREEN_SIZE * (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0])
                goal_start_pos = np.array([wx, wy])
                dragging_goal = True
            
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if dragging_goal and goal_start_pos is not None:
                    mx, my = pygame.mouse.get_pos()
                    # Convert screen to world
                    wx = mx / SCREEN_SIZE * (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0])
                    wy = (SCREEN_SIZE - my) / SCREEN_SIZE * (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0])
                    goal_end_pos = np.array([wx, wy])
                    
                    # Calculate direction from drag vector
                    drag_vector = goal_end_pos - goal_start_pos
                    if np.linalg.norm(drag_vector) > 0.1:  # Minimum drag distance
                        heading = np.arctan2(drag_vector[1], drag_vector[0])
                    else:
                        heading = np.pi/2  # Default heading if no drag
                    
                    x_goal = np.array([goal_start_pos[0], goal_start_pos[1], heading], dtype=float)
                    print(f"New goal: position=({x_goal[0]:.2f}, {x_goal[1]:.2f}), heading={np.degrees(heading):.1f}°")
                
                dragging_goal = False
                goal_start_pos = None
            
            # Right-click and drag to add dynamic obstacle with velocity
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                mx, my = event.pos
                # Convert screen to world
                wx = mx / SCREEN_SIZE * (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0])
                wy = (SCREEN_SIZE - my) / SCREEN_SIZE * (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0])
                obstacle_start_pos = np.array([wx, wy])
                dragging_obstacle = True
            
            if event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                if dragging_obstacle and obstacle_start_pos is not None:
                    mx, my = pygame.mouse.get_pos()
                    # Convert screen to world
                    wx = mx / SCREEN_SIZE * (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0])
                    wy = (SCREEN_SIZE - my) / SCREEN_SIZE * (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0])
                    obstacle_end_pos = np.array([wx, wy])
                    
                    # Calculate velocity from drag vector (scaled down for reasonable speed)
                    drag_vector = obstacle_end_pos - obstacle_start_pos
                    velocity = drag_vector * 0.5  # Scale factor for velocity
                    
                    # Create dynamic obstacle
                    new_obstacle = DynamicObstacle(
                        center=obstacle_start_pos.tolist(),
                        radius=0.7,
                        velocity=velocity.tolist()
                    )
                    dynamic_obstacles.append(new_obstacle)
                    print(f"Added dynamic obstacle at: ({obstacle_start_pos[0]:.2f}, {obstacle_start_pos[1]:.2f}) with velocity: ({velocity[0]:.3f}, {velocity[1]:.3f})")
                
                dragging_obstacle = False
                obstacle_start_pos = None
        # ---------------------------
        # RT-RRT* planning step
        # ---------------------------
        path = planner.step(x_agent, x_goal, obstacles, dynamic_obstacles, rectangles, dt)

        # Move boat along path (if there is at least 1 step ahead)
        boat.dt = dt  # Update boat's timestep to match framerate
        if len(path) >= 1:
            next_node = path[0]
            print("Next node:", next_node.x)
            boat.step(next_node.x)
            x_agent = boat.x[:3]

        # Re-root the tree every second
        # ---------------------------
        # Drawing
        # ---------------------------
        screen.fill(COLOR_BG)

        # Draw environment
        draw_obstacles(screen, obstacles, rectangles)
        draw_dynamic_obstacles(screen, dynamic_obstacles)
        if show_tree:
            draw_tree(screen, planner.tree)
        if show_collision_hulls:
            draw_collision_hulls(screen, path)
        draw_path(screen, path)
        draw_boat(screen, boat)
        draw_goal(screen, x_goal)
        draw_ruler(screen)
        draw_root_node(screen, planner.tree)
        draw_sampling_ellipse(screen, planner.c_best, planner.tree.root.x, x_goal)
        draw_search_radius(screen, planner.tree)
        
        # Draw goal direction preview while dragging
        if dragging_goal and goal_start_pos is not None:
            mx, my = pygame.mouse.get_pos()
            wx = mx / SCREEN_SIZE * (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0])
            wy = (SCREEN_SIZE - my) / SCREEN_SIZE * (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0])
            
            start_screen = world_to_screen(goal_start_pos)
            end_screen = world_to_screen(np.array([wx, wy]))
            
            # Draw arrow from start to current mouse position
            pygame.draw.line(screen, (255, 255, 0), start_screen, end_screen, 3)
            pygame.draw.circle(screen, (255, 255, 0), start_screen, 8, 2)
            
            # Draw arrowhead
            angle = math.atan2(end_screen[1] - start_screen[1], end_screen[0] - start_screen[0])
            arrow_len = 15
            arrow_angle = math.pi / 6
            
            p1 = (end_screen[0] - arrow_len * math.cos(angle - arrow_angle),
                  end_screen[1] - arrow_len * math.sin(angle - arrow_angle))
            p2 = (end_screen[0] - arrow_len * math.cos(angle + arrow_angle),
                  end_screen[1] - arrow_len * math.sin(angle + arrow_angle))
            
            pygame.draw.line(screen, (255, 255, 0), end_screen, p1, 3)
            pygame.draw.line(screen, (255, 255, 0), end_screen, p2, 3)
        
        # Draw dynamic obstacle creation preview while dragging
        if dragging_obstacle and obstacle_start_pos is not None:
            mx, my = pygame.mouse.get_pos()
            wx = mx / SCREEN_SIZE * (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0])
            wy = (SCREEN_SIZE - my) / SCREEN_SIZE * (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0])
            
            start_screen = world_to_screen(obstacle_start_pos)
            end_screen = world_to_screen(np.array([wx, wy]))
            
            # Draw obstacle preview circle
            pygame.draw.circle(screen, (255, 100, 100), start_screen, 8, 2)
            
            # Draw velocity arrow
            if np.linalg.norm(np.array(end_screen) - np.array(start_screen)) > 5:
                pygame.draw.line(screen, (255, 100, 100), start_screen, end_screen, 2)
                
                # Draw arrowhead
                angle = math.atan2(end_screen[1] - start_screen[1], end_screen[0] - start_screen[0])
                arrow_len = 10
                arrow_angle = math.pi / 6
                
                p1 = (end_screen[0] - arrow_len * math.cos(angle - arrow_angle),
                      end_screen[1] - arrow_len * math.sin(angle - arrow_angle))
                p2 = (end_screen[0] - arrow_len * math.cos(angle + arrow_angle),
                      end_screen[1] - arrow_len * math.sin(angle + arrow_angle))
                
                pygame.draw.line(screen, (255, 100, 100), end_screen, p1, 2)
                pygame.draw.line(screen, (255, 100, 100), end_screen, p2, 2)
        
        # Instructions and info overlay
        font = pygame.font.Font(None, 24)
        instructions = [
            "T: Tree | I: Info | C: Collision Hulls | D: Clear Obstacles",
            "Left-drag: Goal | Right-drag: Dynamic Obstacle"
        ]
        for i, text in enumerate(instructions):
            surf = font.render(text, True, (200, 200, 200))
            screen.blit(surf, (10, 10 + i * 25))
        
        # Show planner info
        if show_info:
            num_nodes = len(planner.tree.index.nodes)
            path_length = len(path)
            c_best = planner.c_best if planner.c_best != float('inf') else 'N/A'
            
            info_texts = [
                f"Nodes: {num_nodes}",
                f"Path: {path_length} waypoints",
                f"Best cost: {c_best if isinstance(c_best, str) else f'{c_best:.2f}'}",
                f"FPS: {int(clock.get_fps())}"
            ]
            
            y_offset = SCREEN_SIZE - 120
            for i, text in enumerate(info_texts):
                surf = font.render(text, True, (200, 200, 200))
                screen.blit(surf, (10, y_offset + i * 25))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
