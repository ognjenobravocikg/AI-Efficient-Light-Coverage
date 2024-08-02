import random
import numpy as np
import math
import copy
from itertools import combinations

def generate_room(width, height, obstacle_density):
    room = [[0 for _ in range(width)] for _ in range(height)]

    # Create walls (borders)
    for i in range(width):
        room[0][i] = 255
        room[height - 1][i] = 255
    for i in range(height):
        room[i][0] = 255
        room[i][width - 1] = 255

    # Add obstacles
    for _ in range(int(width * height * obstacle_density)):
        x_index = random.randint(1, height - 2)
        y_index = random.randint(1, width - 2)
        room[x_index][y_index] = 255

    return room

def print_room(room, height, width):
    for i in range(height):
        for j in range(width):
            print(room[i][j], end=' ')
        print('\n')

# Calculation of the light of the room
def percent_light(room):
    light_counter = 0
    blank_counter = 0
    for y in range(len(room)):
        for x in range(len(room[y])):
            if room[y][x] == 64:
                light_counter += 1
            elif room[y][x] == 0:
                blank_counter += 1

    return light_counter / (light_counter + blank_counter) * 100

# Ray casting for the light source, for more realistic lighting 
def cast_light(room, light_x, light_y, radius):
    height = len(room)
    width = len(room[0])
    
    def cast_ray(x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if math.sqrt((x0 - light_x) ** 2 + (y0 - light_y) ** 2) <= radius:
                if room[y0][x0] == 0:
                    room[y0][x0] = 64
            if x0 == x1 and y0 == y1:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            if room[y0][x0] == 255:
                break

    for angle in range(0, 360, 1):
        x1 = light_x + int(radius * math.cos(math.radians(angle)))
        y1 = light_y + int(radius * math.sin(math.radians(angle)))
        cast_ray(light_x, light_y, x1, y1)

    room[light_y][light_x] = 128

# Function to evaluate light coverage for a given set of light positions
def evaluate_light_positions(room, light_positions, radius):
    test_room = copy.deepcopy(room)
    for (x, y) in light_positions:
        cast_light(test_room, x, y, radius)
    return percent_light(test_room)

# Exhaustive search for the optimal spots to put the light sources
def search_optimal_lights(room, width, height, radius, num_lights):
    best_positions = []
    max_percent = 0
    empty_positions = [(x, y) for y in range(height) for x in range(width) if room[y][x] != 255]

    for positions in combinations(empty_positions, num_lights):
        current_percent = evaluate_light_positions(room, positions, radius)
        if current_percent > max_percent:
            max_percent = current_percent
            best_positions = positions

    return best_positions

# Generate the room
room = generate_room(15, 15, 0.1)

# Search for the optimal positions for multiple light sources
num_lights = 2  # Number of light sources
optimal_positions = search_optimal_lights(room, 15, 15, 2, num_lights)

# Cast light in the original room at the optimal locations
print(f'The optimal lighting source is: {optimal_positions}')

