import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def load_data(images_path, positions_path, num_examples, prefix):
    images = []
    positions = []
    for i in range(num_examples):
        img = Image.open(os.path.join(images_path, f'{prefix}_{i}.png')).convert('L')
        img = np.array(img)
        images.append(img)

        with open(os.path.join(positions_path, f'results_{i}.txt')) as f:
            pos = []
            for line in f:
                x, y = line.strip().split(',')
                pos.append([int(x), int(y)])
        positions.append(pos)
    
    return np.array(images), np.array(positions)

def save_predicted_positions(predictions, output_folder, num_examples):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i in range(num_examples):
        with open(os.path.join(output_folder, f'predicted_{i}.txt'), 'w') as f:
            for j in range(len(predictions[i]) // 2):
                f.write(f"{int(predictions[i][2*j])},{int(predictions[i][2*j+1])}\n")

# Paths to training data
images_path = 'training/images'
positions_path = 'training/positions'

# Load training data
num_training_examples = 100
images, positions = load_data(images_path, positions_path, num_training_examples, 'training_example')

# Normalize images
images = images / 255.0
images = images[..., np.newaxis]  # Add channel dimension

# Reshape positions
positions = positions.reshape(num_training_examples, -1)

# Split data into training and validation sets
train_images = images[:80]
train_positions = positions[:80]
val_images = images[80:]
val_positions = positions[80:]

def build_cnn(input_shape, num_outputs):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_outputs)  # Output layer for x and y coordinates
    ])
    return model

num_lights = 2
input_shape = (15, 15, 1)  # Adjust based on your image size
num_outputs = 2 * num_lights  # For x and y coordinates of each light source

model = build_cnn(input_shape, num_outputs)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

model.summary()

# Train the model
model.fit(train_images, train_positions, epochs=200, batch_size=16, validation_data=(val_images, val_positions))

# Save the model
model.save('light_position_cnn_model.keras')

print("Model training completed and saved.")
