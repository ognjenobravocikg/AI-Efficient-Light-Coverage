from training import load_data, save_predicted_positions
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from worker_module import evaluate_light_positions
from training import ScaleLayer  

NUM_OF_TESTING_IMG = 20

# Register the custom layer
custom_objects = {"ScaleLayer": ScaleLayer}

# Load the model with custom objects
model = load_model('light_position_cnn_model.keras', custom_objects=custom_objects)

test_images, test_positions = load_data('test/images', 'test/positions', NUM_OF_TESTING_IMG, 'test_example')
test_images = test_images / 255.0
test_images = test_images[..., np.newaxis]
test_positions = test_positions.reshape(NUM_OF_TESTING_IMG, -1)

# Evaluate the model
loss = model.evaluate(test_images, test_positions)
print(f'Test loss: {loss}')

predicted_positions = model.predict(test_images)
predicted_positions_folder = 'test/predicted_positions'
save_predicted_positions(predicted_positions, predicted_positions_folder, len(test_images))

radius = 10  # Set the light radius

# Evaluate predicted and actual light positions
for i in range(len(test_images)):
    img = Image.open(os.path.join('test/images', f'test_example_{i}.png')).convert('L')
    room = np.array(img)

    # Load actual positions from file
    with open(os.path.join('test/positions', f'results_{i}.txt')) as f:
        actual_positions = [(int(x), int(y)) for x, y in (line.strip().split(',') for line in f)]

    # Load predicted positions from file
    with open(os.path.join(predicted_positions_folder, f'predicted_{i}.txt')) as f:
        predicted_positions = [(int(x), int(y)) for x, y in (line.strip().split(',') for line in f)]

    # Evaluate light coverage for actual and predicted positions
    actual_percent = evaluate_light_positions(actual_positions, room, radius)
    predicted_percent = evaluate_light_positions(predicted_positions, room, radius)

    print(f'Example {i}: Actual Light Coverage: {actual_percent:.2f}%, Predicted Light Coverage: {predicted_percent:.2f}%')
