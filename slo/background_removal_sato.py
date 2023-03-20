from skimage.filters import median, sato, apply_hysteresis_threshold
from skimage.morphology import disk, black_tophat, remove_small_objects, binary_erosion
from skimage.filters.rank import entropy
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

dilated_path = osp.join(output_dir, 'star21_OSC_dilated.jpg')
black_tophat_path = osp.join(output_dir, 'star21_OSC_black_tophat.jpg')
no_small_path = osp.join(output_dir, 'star21_OSC_no_small.jpg')

sato_no_small_path = osp.join(output_dir, 'star21_OSC_sato_no_small.jpg')
sato_low_path = osp.join(output_dir, 'star21_OSC_sato_low.jpg')
sato_high_path = osp.join(output_dir, 'star21_OSC_sato_high.jpg')
sato_hyst_path = osp.join(output_dir, 'star21_OSC_sato_hyst.jpg')

eroded_path = osp.join(output_dir, 'star21_OSC_eroded.jpg')

entropy_path = osp.join(output_dir, 'star21_OSC_sato_entropy.jpg')


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
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
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

# Attempt with dilation
# Perform dilation on the image using the structuring element
dilated_image = cv2.dilate(equalized_image, disk(1))
skimage.io.imsave(dilated_path, dilated_image)

# Attempt with black_tophat
# Better but not quite
footprint = disk(1)
black_tophat_image = black_tophat(equalized_image, footprint)
skimage.io.imsave(black_tophat_path, equalized_image+black_tophat_image)

# Attempt with remove_small_object
# Applying multi-Otsu threshold for the default value, generating
# three classes.
from skimage.filters import threshold_multiotsu
thresholds = threshold_multiotsu(equalized_image, classes=2)
# Using the threshold values, we generate the three regions.
otsu = np.digitize(equalized_image, bins=thresholds).astype(bool) # Ah it would fill the vessels with gaps
otsu = np.invert(otsu) # to get rid of the small dark spots
no_small = remove_small_objects(otsu, 300) # empirical value 
# add back the corner as black
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > ((nrows / 2)**2)-4000) # adds a little margin to avoid white circle
img_mask[invalid_pixels] = 0
no_small = img_mask.astype(bool)*no_small
skimage.io.imsave(no_small_path, no_small)
# best option so far


##########################################################################################
# After that there is some pruning and filling to do

# Attempt 1
sat = sato(no_small, sigmas=range(1, 7, 1), black_ridges=False)
skimage.io.imsave(sato_no_small_path, sat)
# not bad with those parameters
# the unnecessary bits are grayers around the main nerves
# Histeresis could be good for that
# Apply hysteresis threshold
low = 75
high = 100
lowt = (sat*255 > low).astype(int)
hight = (sat*255 > high).astype(int)
hyst = apply_hysteresis_threshold(sat*255, low, high)
# Save the hysteresis images
skimage.io.imsave(sato_low_path, lowt)
skimage.io.imsave(sato_high_path, hight)
skimage.io.imsave(sato_hyst_path, hyst)

# White lines are a bit too thick ?
#eroded = binary_erosion(hyst, disk(1))
#skimage.io.imsave(eroded_path, eroded)

# Attempt 2



##########################################################################################
# Try the entropy 
#entropy_image = entropy(no_small.astype('uint8'), disk(1), mask = img_mask)
#skimage.io.imsave(entropy_path, entropy_image)
# Didn't expect that, what to do now ??