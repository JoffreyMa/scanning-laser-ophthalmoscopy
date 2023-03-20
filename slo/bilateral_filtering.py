import skimage.restoration
import skimage.io
from skimage import util
from os import path as osp
import numpy as np

input_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/input/images_IOSTAR'
output_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/output'
image_path = osp.join(input_dir, 'star01_OSC.jpg')

bilateral_path = osp.join(output_dir, 'star01_OSC_bilateral.jpg')

# Read the image
image = skimage.io.imread(image_path)

# Apply bilateral filtering
filtered_image = skimage.restoration.denoise_bilateral(image, sigma_color=0.025, sigma_spatial=10)

# Save the filtered image
skimage.io.imsave(bilateral_path, filtered_image)
