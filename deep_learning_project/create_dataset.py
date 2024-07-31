import os
import numpy as np
from PIL import Image
from create_room import generate_room, search_optimal_lights

# Function for saving the images
def save_room_image(room, filename):
    room_array = np.array(room, dtype=np.uint8)  # Convert the list to a NumPy array
    img = Image.fromarray(room_array, 'L')  # 'L' mode for grayscale images
    img.save(filename)
    print(f'Image saved as {filename}')

# Function for saving the optimal light positions
def save_optimal_positions(positions, filename):
    with open(filename, 'w') as f:
        for pos in positions:
            f.write(f"{pos[0]}, {pos[1]}\n")
    print(f'Optimal positions saved as {filename}')

# Hyperparameters
image_width = 15
image_height = 15
wall_density = 0.1
num_lights = 2
ray_spread = 2

# Ensuring the same directories hierarchy
training_path = 'training'
test_path = 'test'

if not os.path.exists(training_path):
    os.makedirs(training_path)
    os.makedirs(os.path.join(training_path, 'images'))
    os.makedirs(os.path.join(training_path, 'positions'))

# Creating the training dataset
for training_example in range(100):
    room = generate_room(image_height, image_width, wall_density)
    
    image_filename = os.path.join(training_path, 'images', f'training_example_{training_example}.png')
    save_room_image(room, image_filename)

    optimal_lights = search_optimal_lights(room, image_height, image_width, ray_spread, num_lights)
    
    positions_filename = os.path.join(training_path, 'positions', f'results_{training_example}.txt')
    save_optimal_positions(optimal_lights, positions_filename)

if not os.path.exists(test_path):
    os.makedirs(test_path)
    os.makedirs(os.path.join(test_path, 'images'))
    os.makedirs(os.path.join(test_path, 'positions'))

# Creating the test dataset
for test_example in range(30):
    room = generate_room(image_height, image_width, wall_density)
    
    image_filename = os.path.join(test_path, 'images', f'test_example_{test_example}.png')
    save_room_image(room, image_filename)

    optimal_lights = search_optimal_lights(room, image_height, image_width, ray_spread, num_lights)
    
    positions_filename = os.path.join(test_path, 'positions', f'results_{test_example}.txt')
    save_optimal_positions(optimal_lights, positions_filename)
