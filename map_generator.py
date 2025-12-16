"""
Map Generator GUI for creating obstacle maps.
Press 1 for circle mode, 2 for rectangle mode.
Click and drag to create obstacles.
Press ENTER to save the map.
Press DELETE to remove last obstacle.
Press C to clear all obstacles.
"""

import pygame
import numpy as np
import json
from config import WORLD_BOUNDS

# Pygame setup
SCREEN_SIZE = 800
FPS = 60

# Colors
COLOR_BG = (25, 25, 25)
COLOR_GRID = (40, 40, 40)
COLOR_OBSTACLE = (200, 60, 60)
COLOR_PREVIEW = (200, 60, 60, 100)
COLOR_TEXT = (200, 200, 200)
COLOR_MODE_CIRCLE = (100, 200, 100)
COLOR_MODE_RECT = (100, 100, 200)

# Modes
MODE_CIRCLE = 1
MODE_RECTANGLE = 2

def world_to_screen(x):
    """Map world coords to screen coords."""
    sx = int((x[0] - WORLD_BOUNDS[0, 0]) /
             (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0]) * SCREEN_SIZE)
    sy = int((x[1] - WORLD_BOUNDS[1, 0]) /
             (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0]) * SCREEN_SIZE)
    sy = SCREEN_SIZE - sy
    return sx, sy

def screen_to_world(sx, sy):
    """Map screen coords to world coords."""
    wx = (sx / SCREEN_SIZE) * (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0]) + WORLD_BOUNDS[0, 0]
    wy = ((SCREEN_SIZE - sy) / SCREEN_SIZE) * (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0]) + WORLD_BOUNDS[1, 0]
    return wx, wy

def draw_grid(screen):
    """Draw background grid."""
    grid_size = 5.0  # meters
    
    for i in np.arange(WORLD_BOUNDS[0, 0], WORLD_BOUNDS[0, 1] + grid_size, grid_size):
        x = (i - WORLD_BOUNDS[0, 0]) / (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0]) * SCREEN_SIZE
        pygame.draw.line(screen, COLOR_GRID, (x, 0), (x, SCREEN_SIZE), 1)
    
    for i in np.arange(WORLD_BOUNDS[1, 0], WORLD_BOUNDS[1, 1] + grid_size, grid_size):
        y = (i - WORLD_BOUNDS[1, 0]) / (WORLD_BOUNDS[1, 1] - WORLD_BOUNDS[1, 0]) * SCREEN_SIZE
        pygame.draw.line(screen, COLOR_GRID, (0, SCREEN_SIZE - y), (SCREEN_SIZE, SCREEN_SIZE - y), 1)

def draw_circle_obstacle(screen, center, radius, color=COLOR_OBSTACLE, width=0):
    """Draw a circular obstacle."""
    cx, cy = world_to_screen(center)
    r = int(radius / (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0]) * SCREEN_SIZE)
    pygame.draw.circle(screen, color, (cx, cy), r, width=width)

def draw_rectangle_obstacle(screen, corner1, corner2, color=COLOR_OBSTACLE, width=0):
    """Draw a rectangular obstacle."""
    x1, y1 = world_to_screen(corner1)
    x2, y2 = world_to_screen(corner2)
    
    left = min(x1, x2)
    top = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    
    pygame.draw.rect(screen, color, (left, top, w, h), width=width)

def save_map(obstacles, filename="map_obstacles.json"):
    """Save obstacles to JSON file."""
    data = {
        'circles': [],
        'rectangles': []
    }
    
    for obs in obstacles:
        if obs['type'] == 'circle':
            data['circles'].append({
                'center': [float(obs['center'][0]), float(obs['center'][1])],
                'radius': float(obs['radius'])
            })
        elif obs['type'] == 'rectangle':
            data['rectangles'].append({
                'corner1': [float(obs['corner1'][0]), float(obs['corner1'][1])],
                'corner2': [float(obs['corner2'][0]), float(obs['corner2'][1])]
            })
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Map saved to {filename}")
    print(f"  Circles: {len(data['circles'])}")
    print(f"  Rectangles: {len(data['rectangles'])}")

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Map Generator - Press 1 (Circle) or 2 (Rectangle)")
    clock = pygame.time.Clock()
    
    # State
    obstacles = []
    mode = MODE_CIRCLE
    dragging = False
    start_pos = None
    current_pos = None
    
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Keyboard controls
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    mode = MODE_CIRCLE
                    print("Mode: Circle")
                elif event.key == pygame.K_2:
                    mode = MODE_RECTANGLE
                    print("Mode: Rectangle")
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    save_map(obstacles)
                elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                    if obstacles:
                        removed = obstacles.pop()
                        print(f"Removed {removed['type']}")
                elif event.key == pygame.K_c:
                    obstacles.clear()
                    print("Cleared all obstacles")
            
            # Mouse controls
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                wx, wy = screen_to_world(mx, my)
                start_pos = np.array([wx, wy])
                dragging = True
            
            if event.type == pygame.MOUSEMOTION:
                if dragging:
                    mx, my = event.pos
                    wx, wy = screen_to_world(mx, my)
                    current_pos = np.array([wx, wy])
            
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if dragging and start_pos is not None and current_pos is not None:
                    if mode == MODE_CIRCLE:
                        center = start_pos
                        radius = np.linalg.norm(current_pos - start_pos)
                        if radius > 0.1:  # Minimum radius
                            obstacles.append({
                                'type': 'circle',
                                'center': center,
                                'radius': radius
                            })
                            print(f"Added circle: center={center}, radius={radius:.2f}")
                    elif mode == MODE_RECTANGLE:
                        if not np.allclose(start_pos, current_pos):
                            obstacles.append({
                                'type': 'rectangle',
                                'corner1': start_pos,
                                'corner2': current_pos
                            })
                            print(f"Added rectangle: corners={start_pos}, {current_pos}")
                
                dragging = False
                start_pos = None
                current_pos = None
        
        # Drawing
        screen.fill(COLOR_BG)
        draw_grid(screen)
        
        # Draw existing obstacles
        for obs in obstacles:
            if obs['type'] == 'circle':
                draw_circle_obstacle(screen, obs['center'], obs['radius'])
            elif obs['type'] == 'rectangle':
                draw_rectangle_obstacle(screen, obs['corner1'], obs['corner2'])
        
        # Draw preview
        if dragging and start_pos is not None and current_pos is not None:
            preview_surface = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE), pygame.SRCALPHA)
            
            if mode == MODE_CIRCLE:
                radius = np.linalg.norm(current_pos - start_pos)
                cx, cy = world_to_screen(start_pos)
                r = int(radius / (WORLD_BOUNDS[0, 1] - WORLD_BOUNDS[0, 0]) * SCREEN_SIZE)
                pygame.draw.circle(preview_surface, COLOR_PREVIEW, (cx, cy), r)
                pygame.draw.circle(preview_surface, COLOR_OBSTACLE, (cx, cy), r, 2)
                
                # Draw radius line
                end_x, end_y = world_to_screen(current_pos)
                pygame.draw.line(preview_surface, (255, 255, 255, 150), (cx, cy), (end_x, end_y), 2)
            
            elif mode == MODE_RECTANGLE:
                x1, y1 = world_to_screen(start_pos)
                x2, y2 = world_to_screen(current_pos)
                
                left = min(x1, x2)
                top = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                
                pygame.draw.rect(preview_surface, COLOR_PREVIEW, (left, top, w, h))
                pygame.draw.rect(preview_surface, COLOR_OBSTACLE, (left, top, w, h), 2)
            
            screen.blit(preview_surface, (0, 0))
        
        # Draw UI
        font = pygame.font.Font(None, 28)
        
        # Mode indicator
        mode_text = "Mode: CIRCLE (1)" if mode == MODE_CIRCLE else "Mode: RECTANGLE (2)"
        mode_color = COLOR_MODE_CIRCLE if mode == MODE_CIRCLE else COLOR_MODE_RECT
        surf = font.render(mode_text, True, mode_color)
        screen.blit(surf, (10, 10))
        
        # Instructions
        font_small = pygame.font.Font(None, 20)
        instructions = [
            "Click & Drag: Create obstacle",
            "1/2: Switch mode",
            "ENTER: Save map",
            "DELETE: Remove last",
            "C: Clear all"
        ]
        
        for i, text in enumerate(instructions):
            surf = font_small.render(text, True, COLOR_TEXT)
            screen.blit(surf, (10, 50 + i * 22))
        
        # Obstacle count
        count_text = f"Obstacles: {len(obstacles)}"
        surf = font_small.render(count_text, True, COLOR_TEXT)
        screen.blit(surf, (10, SCREEN_SIZE - 30))
        
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()
