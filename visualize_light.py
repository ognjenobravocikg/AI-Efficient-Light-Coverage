import os
import numpy as np
from PIL import Image
from worker_module import cast_light

images_path = 'test/images'
predicted_positions_path = 'test/predicted_positions'
output_path = 'test/light_visualizations'

if not os.path.exists(output_path):
    os.makedirs(output_path)

NUM_OF_TESTING_IMG = 19
light_radius = 10 

# uses cast_light from the worker_module to cast lights using ray-casting 
#TAKES: image, position, light_radius
#RETURNS: room with the cast light
def visualize_light_placement(image, positions, light_radius):
    room = np.array(image)
    for x, y in positions:
        cast_light(room, x, y, light_radius)
    return room

# for loop for casting the light, takes an image, it's appropriate position for efficient light coverage and then it performs the operation
for i in range(NUM_OF_TESTING_IMG):
    img = Image.open(os.path.join(images_path, f'test_example_{i}.png')).convert('L')
    
    positions = []
    with open(os.path.join(predicted_positions_path, f'predicted_{i}.txt')) as f:
        for line in f:
            x, y = line.strip().split(',')
            positions.append((int(x), int(y)))

    light_room = visualize_light_placement(img, positions, light_radius)
    
    output_img = Image.fromarray(light_room)
    output_img.save(os.path.join(output_path, f'light_visualization_{i}.png'))

print("Light placement visualizations saved.")
