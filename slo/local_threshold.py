# could get completely rid of the background simply
import skimage.io
from skimage.morphology import disk, black_tophat, remove_small_objects
from skimage.filters import threshold_local, sato

import numpy as np
from os import path as osp

input_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/input/images_IOSTAR'
output_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/output'
image_path = osp.join(input_dir, 'star01_OSC.jpg')

local_path = osp.join(output_dir, 'star01_OSC_local.jpg')
black_local_path = osp.join(output_dir, 'star01_OSC_local_black.jpg')
local_no_small_path = osp.join(output_dir, 'star01_OSC_local_no_small.jpg')

# Read the image
image = skimage.io.imread(image_path, as_gray=True)

# Alternatively consider pixels in inner disk only
nrows, ncols = image.shape
row, col = np.ogrid[:nrows, :ncols]
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
image[invalid_pixels] = 255

gaussian = image > threshold_local(image, block_size=21, method='gaussian')
gaussian[invalid_pixels] = 0
skimage.io.imsave(local_path, gaussian)

# Black tophat
footprint = disk(1)
black_tophat_image = black_tophat(gaussian, footprint)
skimage.io.imsave(black_local_path, gaussian+black_tophat_image)

# No small objects
from skimage.filters import threshold_multiotsu
thresholds = threshold_multiotsu(gaussian+black_tophat_image, classes=2)
# Using the threshold values, we generate the three regions.
otsu = np.digitize(gaussian+black_tophat_image, bins=thresholds).astype(bool) # Ah it would fill the vessels with gaps
otsu = np.invert(otsu) # to get rid of the small dark spots
no_small = remove_small_objects(otsu, 100) # empirical value 
skimage.io.imsave(local_no_small_path, no_small)