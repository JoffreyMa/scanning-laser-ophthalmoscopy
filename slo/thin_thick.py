# Goal here is to try out CLAHE, and split detection of wide and thin vessels

from skimage.filters import median, meijering, apply_hysteresis_threshold,threshold_otsu
from skimage.morphology import disk, black_tophat, remove_small_objects, binary_erosion, area_opening
from skimage.filters.rank import entropy
from skimage.exposure import equalize_adapthist, equalize_hist
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import skimage.io
from skimage import util
from os import path as osp
import numpy as np
import cv2
import pywt

input_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/input/images_IOSTAR'
output_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/output'
image_path = osp.join(input_dir, 'star21_OSC.jpg')

clahe_path = osp.join(output_dir, 'star21_OSC_clahe.jpg')
clahe_black_path = osp.join(output_dir, 'star21_OSC_clahe_black.jpg')
clahe_black_op_path = osp.join(output_dir, 'star21_OSC_clahe_black_op.jpg')

thin_path = osp.join(output_dir, 'star21_OSC_thin.jpg')
thick_path = osp.join(output_dir, 'star21_OSC_thick.jpg')

thin_thick_path = osp.join(output_dir, 'star21_OSC_thin_thick.jpg')
tt_low_path = osp.join(output_dir, 'star21_OSC_thin_thick_low.jpg')
tt_high_path= osp.join(output_dir, 'star21_OSC_thin_thick_high.jpg')
tt_hyst_path= osp.join(output_dir, 'star21_OSC_thin_thick_hyst.jpg')
tt_cleaned_path= osp.join(output_dir, 'star21_OSC_thin_thick_cleand.jpg')

# Read the image
image = skimage.io.imread(image_path, as_gray=True)

clahe = equalize_adapthist(image, kernel_size=64, clip_limit=0.03, nbins=64)
skimage.io.imsave(clahe_path, clahe)

footprint = disk(8)
clahe_black = black_tophat(clahe, footprint)
skimage.io.imsave(clahe_black_path, clahe_black)

clahe_black_op = area_opening(clahe_black, area_threshold=8, connectivity=8, parent=None, tree_traverser=None)
skimage.io.imsave(clahe_black_op_path, clahe_black_op)

# Thick vessels
H_thick = hessian_matrix(clahe_black_op, sigma=4, mode='constant', cval=0, order='rc', use_gaussian_derivatives=False)
thick = 1-hessian_matrix_eigvals(H_thick)[1]
# Global Otsu Thresholding
thick_otsu = thick > threshold_otsu(thick)
skimage.io.imsave(thick_path, thick_otsu)

# Thin vessels
H_thin = hessian_matrix(clahe_black_op, sigma=1, mode='constant', cval=0, order='rc', use_gaussian_derivatives=False)
thin = 1-hessian_matrix_eigvals(H_thin)[1] # 0 corresponds more to contours
skimage.io.imsave(thin_path, thin) # cannot do otsu on thin vessels, it gets too noisy


# Equalize
# needs the mask
# Does not seem to work very well...
nrows, ncols = image.shape
row, col = np.ogrid[:nrows, :ncols]
img_mask = np.ones(image.shape)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
img_mask[invalid_pixels] = 0

skimage.io.imsave(thick_path, thick_otsu)
skimage.io.imsave(thin_path, thin)
thick_otsu_eq = equalize_hist(thick_otsu, nbins=128, mask=img_mask)
skimage.io.imsave(thick_path, thick_otsu_eq)


# Fusion 
thin_thick = thick_otsu_eq+thin # not between 0 and 1 but it's taken care of by skimage
skimage.io.imsave(thin_thick_path, thin_thick)
# There might be too much of a difference between thick and thin.
# And the thin vessels are less visible

# Wavelet fusion
coeffs1 = pywt.wavedec2(thick_otsu_eq, 'db1')
coeffs2 = pywt.wavedec2(thin, 'db1')
fused_coeffs = []
for c1, c2 in zip(coeffs1, coeffs2):
    fused_c = []
    for band1, band2 in zip(c1, c2):
        band_fused = (0.5*band1 + 2*band2)/2
        fused_c.append(band_fused)
    fused_coeffs.append(tuple(fused_c))
fused_image = pywt.waverec2(fused_coeffs, 'db1')
fused_image_min = np.min(fused_image)
fused_image_max = np.max(fused_image)
output = 255* ((fused_image - fused_image_min)/(fused_image_max-fused_image_min))
output = cv2.resize(output,thick_otsu_eq.T.shape)
skimage.io.imsave(thin_thick_path, output)


# With hysterisis I get the thick and the enough of the thin linked to it
tt = output # tt for thick_thin
low = 90
high = 110
lowt = (tt > low).astype(int)
hight = (tt > high).astype(int)
hyst = apply_hysteresis_threshold(tt, low, high)
# Save the hysteresis images
skimage.io.imsave(tt_low_path, lowt)
skimage.io.imsave(tt_high_path, hight)
skimage.io.imsave(tt_hyst_path, hyst)

# Small elements to remove
no_small = remove_small_objects(hyst, 20) # empirical value 
skimage.io.imsave(tt_cleaned_path, no_small)