from skimage.filters import median, meijering, apply_hysteresis_threshold
from skimage.morphology import disk, black_tophat, remove_small_objects, binary_erosion
import skimage.io
from skimage import util
from os import path as osp
import numpy as np
import cv2

input_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/input/images_IOSTAR'
output_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/output'
image_path = osp.join(input_dir, 'star21_OSC.jpg')

median_path = osp.join(output_dir, 'star21_OSC_median.jpg')
uniform_background_path = osp.join(output_dir, 'star21_OSC_unif_back.jpg')
equalized_path = osp.join(output_dir, 'star21_OSC_equalized.jpg')

black_tophat_path = osp.join(output_dir, 'star21_OSC_black_tophat.jpg')

black_low_path = osp.join(output_dir, 'star21_OSC_black_low.jpg')
black_high_path = osp.join(output_dir, 'star21_OSC_black_high.jpg')
black_hyst_path = osp.join(output_dir, 'star21_OSC_black_hyst.jpg')

no_small_path = osp.join(output_dir, 'star21_OSC_black_no_small.jpg')

# Read the image
image = skimage.io.imread(image_path, as_gray=True)

# Apply bilateral filtering
# with the following we get a decent blur
filtered_image = median(image, disk(7))


# Save the filtered image
skimage.io.imsave(median_path, filtered_image)

# Even if blurred we still get those regions of white and gray
# It would be nice to have a very homogenous background
# I can take the overly blurred image which does not take 
# the nerves to much into account
# Now we remove the filtered_image from the image

# invert the background
background_inverse = 255 - filtered_image

# Alternatively consider pixels in inner disk only
nrows, ncols = background_inverse.shape
row, col = np.ogrid[:nrows, :ncols]
img_mask = np.ones(background_inverse.shape)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2 - 1000)
img_mask[invalid_pixels] = 0

background_inverse_disk = img_mask * background_inverse

# Remove the background
# On a side note local thresholding could be good too
uniform_background_image = np.clip(image+background_inverse_disk, 0, 255)
uniform_background_image = uniform_background_image.astype(np.uint8)

skimage.io.imsave(uniform_background_path, uniform_background_image)

# That looks fine but it lacks contrast
# Augment the contrast using histogram equalization
equalized_image = skimage.exposure.equalize_hist(uniform_background_image, mask=img_mask)

# Save the resulting image
skimage.io.imsave(equalized_path, (equalized_image * 255).astype('uint8'))


##########################################################################################
# there is now a lot of small pollution
# let's try dilating to remove those stains

# Attempt with black_tophat
# Better but not quite
footprint = disk(1)
black_tophat_image = black_tophat(equalized_image, footprint)
black = util.invert(equalized_image+black_tophat_image)
black[invalid_pixels] = 0
skimage.io.imsave(black_tophat_path, black)

# Might actually be good enough to perform hysterisis
low = 150
high = 200
lowt = (black*255 > low).astype(int)
hight = (black*255 > high).astype(int)
hyst = apply_hysteresis_threshold(black*255, low, high)
# Save the hysteresis images
skimage.io.imsave(black_low_path, lowt)
skimage.io.imsave(black_high_path, hight)
skimage.io.imsave(black_hyst_path, hyst)

# Little white dots to remove
no_small = remove_small_objects(hyst, 75) # empirical value 
skimage.io.imsave(no_small_path, no_small)