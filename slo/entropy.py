import skimage.io
from skimage.filters.rank import entropy
from skimage.morphology import rectangle, disk
from skimage.filters import threshold_multiotsu

from os import path as osp

input_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/input/images_IOSTAR'
output_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/output'
image_path = osp.join(input_dir, 'star01_OSC.jpg')

entropy_path = osp.join(output_dir, 'star01_OSC_entropy.jpg')

# Read the image
image = skimage.io.imread(image_path, as_gray=True)

# Otsu allow us to avoid the black corners
t0, t1 = threshold_multiotsu(image, classes=3)
mask = (image > t0)

img_entropy = entropy(image, footprint=disk(3), mask = mask)

# I have no idea what to do with that

# Save the otsu image
skimage.io.imsave(entropy_path, img_entropy)