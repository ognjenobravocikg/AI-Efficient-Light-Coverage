import numpy as np
import math

from multiprocessing import Pool, cpu_count
from random import randint
from itertools import combinations


#Generate a maze-like room, white pixels are the walls, black pixels are the blank spaces, image is a 2D array
#TAKES: rows, columns of the 2D matrix or picture and the obstacle_density how many walls should the room have
#RETURNS: generated 2D array or room
def generate_room(rows, columns, obstacle_density):
    room = np.zeros((rows, columns))

    # Create walls (borders)
    room[:, 0] = 255
    room[:, columns-1] = 255
    room[0, :] = 255
    room[rows-1, :] = 255

    # Add obstacles
    for _ in range(int(rows * columns * obstacle_density)):
        y_index = randint(1, rows - 2)
        x_index = randint(1, columns - 2)
        room[y_index][x_index] = 255

    return room

#Calculation of the light of the room, light is represented as a value 64 or a gray hue
#TAKES: 2D room 
#RETURNS: a percentage value
def percent_light(room : np.array):
    flatten_room = room.ravel()

    light_counter = lambda x: np.sum(x == 64)
    blank_counter = lambda x: np.sum(x == 0)
    
    counted_light = light_counter(flatten_room)
    counted_blank = blank_counter(flatten_room)

    return counted_light / (counted_light + counted_blank) * 100

# Ray casting for the light source, for more realistic lighting
#TAKES: room 2D array, position of the light on the x axis, position of the light on the y axis, radius of the light
#RETURNS: a light room, light represented as a 64 value or a gray hue
def cast_light(room, light_x, light_y, radius):
    rows = len(room)
    columns = len(room[0])

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

#a helper function for finding optimal light in room
#TAKES: light_position a touple of x and y coordinates, room a 2D array, radius of the light
#RETURNS: percent of the light room  
def evaluate_light_positions(light_positions, room, radius):
    test_room = room.copy()
    for (x, y) in light_positions:
        cast_light(test_room, x, y, radius)
    return percent_light(test_room)

#a helper function for find optimal light in room
#TAKES room, comb, radius a touple
#RETURNS evaluated percentage value
def worker(params):
    room, comb, radius = params
    return evaluate_light_positions(light_positions=comb, room=room, radius=radius)

# Exhaustive search for the optimal spots to put the light sources using multiproccesing
#TAKES: room a 2D array, radius of light and the number of lights
#RETURNS: best position of the x and y coordinates to light a room
def search_optimal_lights_multi(room, radius, num_lights):

    best_positions = []
    max_percent = 0
    empty_positions = np.argwhere(room == 0)
    all_combinations = combinations(empty_positions, num_lights)
    params = [(room, comb, radius) for comb in all_combinations]

    num_processes = cpu_count()
    pool = Pool(processes=num_processes)
    async_results = [pool.apply_async(worker, (param,)) for param in params]
    results = [result.get() for result in async_results]

    max_percent = max(results)
    best_positions_indexes = results.index(max_percent)
    best_positions = list(combinations(empty_positions, num_lights))[best_positions_indexes]

    pool.close()
    pool.join()

    return best_positions