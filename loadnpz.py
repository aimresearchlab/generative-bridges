import numpy as np
import os
from PIL import Image

# Define paths
file_path = r'C:\Users\Sean\generative-bridges\improved-diffusion\trained_models\samples_1000x32x32x3.npz'
output_dir = r'C:\Users\Sean\generative-bridges\improved-diffusion\sampled_images'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the .npz file
data = np.load(file_path)

# Extract images (saved as 'arr_0')
images = data['arr_0']  # Shape: (100, 32, 32, 3), uint8 [0, 255]

# Save all images to the output directory
for i in range(len(images)):
    img = Image.fromarray(images[i])
    img.save(os.path.join(output_dir, f'image_{i+1:03d}.png'))

print(f"Saved {len(images)} images to {output_dir}")

# Optionally, extract labels if present (saved as 'arr_1' for class-conditional models)
if 'arr_1' in data:
    labels = data['arr_1']
    print("Labels for first 5 images:", labels[:5])