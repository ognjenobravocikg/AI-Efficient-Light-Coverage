import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense
from PIL import Image
import os
from tensorflow.keras.optimizers import Adam

# Load the data
def load_data(images_path, positions_path, num_examples):
    images = []
    positions = []
    for i in range(num_examples):
        img = Image.open(os.path.join(images_path, f'training_example_{i}.png')).convert('L')
        img = np.array(img)
        images.append(img)

        with open(os.path.join(positions_path, f'results_{i}.txt')) as f:
            pos = []
            for line in f:
                x, y = line.strip().split(',')
                pos.append([int(x), int(y)])
        positions.append(pos)
    
    return np.array(images), np.array(positions)

# Paths to training data
images_path = 'training/images'
positions_path = 'training/positions'

# Load training data
num_training_examples = 100
images, positions = load_data(images_path, positions_path, num_training_examples)

# U-Net architecture
def build_unet(input_shape, num_lights):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    # Output layer for regression
    f = Flatten()(c9)
    outputs = Dense(2 * num_lights)(f)

    model = Model(inputs, outputs)
    return model

# Hyperparameters
input_shape = (15, 15, 1)
num_lights = 2

# Build and compile the model
model = build_unet(input_shape, num_lights)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Load your data
# Assuming `images` and `positions` are loaded as before
images = images / 255.0
images = images[..., np.newaxis]  # Add channel dimension

# Train the model
model.fit(images, positions.reshape(num_training_examples, -1), epochs=3, batch_size=16, validation_split=0.2)

# Save the model
model.save('light_position_unet_model.h5')

print("Model training completed and saved.")
