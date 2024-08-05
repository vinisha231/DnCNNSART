import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import Subtract
from glob import glob

def DnCNN(depth=17, filters=64, image_channels=1, use_bn=True):
    input_layer = Input(shape=(None, None, image_channels), name='input')
    x = layers.Conv2D(filters, kernel_size=3, padding='same', activation='relu')(input_layer)

    for _ in range(depth-2):
        x = layers.Conv2D(filters=filters, kernel_size=3, padding='same')(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    output_layer = layers.Conv2D(filters=image_channels, kernel_size=3, padding='same')(x)
    output_layer = Subtract()([input_layer, output_layer])
    model = models.Model(inputs=input_layer, outputs=output_layer, name='DnCNN')
    return model

def read_flt_file(file_path, shape):
    with open(file_path, 'rb') as f:
        data = f.read()
        return np.array(struct.unpack('f' * (len(data) // 4), data)).reshape(shape)

def load_real_data(noisy_dir, normal_dir, shape, limit=100):
    noisy_paths = sorted(glob(os.path.join(noisy_dir, '*.flt')))[:limit]
    normal_paths = sorted(glob(os.path.join(normal_dir, '*.flt')))[:limit]

    noisy_images = [read_flt_file(path, shape) for path in noisy_paths]
    normal_images = [read_flt_file(path, shape) for path in normal_paths]

    noisy_images = np.array(noisy_images)
    normal_images = np.array(normal_images)

    return noisy_images, normal_images

def save_flt_file(file_path, data):
    with open(file_path, 'wb') as f:
        f.write(struct.pack('f' * data.size, *data.flatten()))

def display_images(original, noisy, denoised):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Noisy Image')
    plt.imshow(noisy, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Denoised Image')
    plt.imshow(denoised, cmap='gray')
    plt.axis('off')

    plt.show()

# Define input and output directories and image shape
#noisy_dir = '/mmfs1/gscratch/uwb/CT_images/recons2024/60views'
noisy_dir = '/mmfs1/gscratch/uwb/limitedangles/limitedangleimages'
normal_dir = '/mmfs1/gscratch/uwb/CT_images/recons2024/900views'
output_dir = '/mmfs1/gscratch/uwb/vdhaya/output'
os.makedirs(output_dir, exist_ok=True)
image_shape = (512, 512)

# Load real data
X_train, y_train = load_real_data(noisy_dir, normal_dir, image_shape)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Create the model
model = DnCNN(depth=17, filters=64, image_channels=1, use_bn=True)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=250, batch_size=8, verbose=1)
model.save('dncnn_model.h5')
model.save_weights('dncnn_model.weights.h5')

# Print model summary
model.summary()
#noisy_dir2 = '/mmfs1/gscratch/uwb/vdhaya/meansquarelimitedloss/output' 
noisy_dir2 = '/mmfs1/gscratch/uwb/limitedangles/limitedangletest'
# Process each image in the noisy directory
for i, image_path in enumerate(sorted(glob(os.path.join(noisy_dir2, '*.flt'))[:100])):
    # Preprocess the test image
    test_image = read_flt_file(image_path, image_shape)
    test_image = np.expand_dims(test_image, axis=(0, -1))  # Add batch and channel dimensions

    # Denoise the grayscale image using the DnCNN model
    denoised_image = model.predict(test_image)

    # Save the denoised image to the specified folder
    output_image_path = os.path.join(output_dir, f'denoised_image_{i + 1}.flt')
    save_flt_file(output_image_path, denoised_image[0, :, :, 0])
    print(f"Denoised image saved to: {output_image_path}")

    # Optionally display the images
    display_images(test_image[0, :, :, 0], test_image[0, :, :, 0], denoised_image[0, :, :, 0])
