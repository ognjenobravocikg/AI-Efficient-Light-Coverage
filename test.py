from training import load_data, save_predicted_positions
import numpy as np
from PIL import Image
import os
import copy
import math
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model
model = load_model('light_position_cnn_model.keras')

# Load test data
test_images, test_positions = load_data('test/images', 'test/positions', 30, 'test_example')
test_images = test_images / 255.0
test_images = test_images[..., np.newaxis]
test_positions = test_positions.reshape(30, -1)

# Evaluate the model
loss = model.evaluate(test_images, test_positions)
print(f'Test loss: {loss}')

# Predict positions
predicted_positions = model.predict(test_images)

# Save predicted positions
predicted_positions_folder = 'test/predicted_positions'
save_predicted_positions(predicted_positions, predicted_positions_folder, len(test_images))

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

def evaluate_positions(room, positions, radius):
    room_copy = copy.deepcopy(room)
    for (x, y) in positions:
        cast_light(room_copy, x, y, radius)
    return percent_light(room_copy)

radius = 2  # Set the light radius

# Evaluate predicted and actual light positions
for i in range(len(test_images)):
    img = Image.open(os.path.join('test/images', f'test_example_{i}.png')).convert('L')
    room = np.array(img)

    with open(os.path.join('test/positions', f'results_{i}.txt')) as f:
        actual_positions = [(int(x), int(y)) for x, y in (line.strip().split(',') for line in f)]

    with open(os.path.join(predicted_positions_folder, f'predicted_{i}.txt')) as f:
        predicted_positions = [(int(x), int(y)) for x, y in (line.strip().split(',') for line in f)]

    actual_percent = evaluate_positions(room, actual_positions, radius)
    predicted_percent = evaluate_positions(room, predicted_positions, radius)

    print(f'Example {i}: Actual Light Coverage: {actual_percent:.2f}%, Predicted Light Coverage: {predicted_percent:.2f}%')
