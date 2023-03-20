# File containing all the tested process for this project
# all process functions must take an uint8 grayscale image as input and return an img_out

import cv2
from skimage.filters import threshold_multiotsu, sato, apply_hysteresis_threshold, meijering
from skimage.filters import median
from skimage.morphology import disk, remove_small_objects, binary_erosion, black_tophat
from skimage.exposure import equalize_hist
from skimage import util
import numpy as np

def apply_denoise_gaussian_canny(img):
    # Denoise
    img_denoised = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=7)
    # Blur before Canny
    img_blur = cv2.GaussianBlur(img_denoised, ksize=(7,7), sigmaX=2)
    # Canny edge detection
    img_out = cv2.Canny(image=img_blur, threshold1=0, threshold2=8000, apertureSize=7)
    return img_out

def apply_sato_hysteresis(img):
    # Otsu to remove black corners
    t0, _ = threshold_multiotsu(img, classes=3)
    mask = (img > t0)
    # Sato filter to detect linked tubes
    vessels = sato(img, sigmas=range(2, 3)) * mask
    # Apply histeresis threshold
    # we get large vessels and theirs smaller connections
    low = 3
    high = 8
    hyst = apply_hysteresis_threshold(vessels, low, high)
    return hyst

def apply_background_removal(img):
    # Apply bilateral filtering
    # with the following we get a decent blur
    filtered_image = median(img, disk(7))
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
    uniform_background_image = np.clip(img+background_inverse_disk, 0, 255)
    uniform_background_image = uniform_background_image.astype(np.uint8)

    # That looks fine but it lacks contrast
    # Augment the contrast using histogram equalization
    equalized_image = equalize_hist(uniform_background_image, mask=img_mask)

    return equalized_image

def apply_background_removal_filtered(img):
    # Alternatively consider pixels in inner disk only
    nrows, ncols = img.shape
    row, col = np.ogrid[:nrows, :ncols]
    img_mask = np.ones(img.shape)
    equalized_image = apply_background_removal(img)

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

    return no_small

def apply_background_removal_meijering(img):
    no_small = apply_background_removal_filtered(img)

    # After that there is some pruning and filling to do
    # meijering looks good to do both at the same time
    meije = meijering(no_small, sigmas=[4], alpha=0.7, black_ridges=False) 
    # not bad with those parameters
    # the unnecessary bits are grayers around the main nerves
    # Histeresis could be good for that
    # Apply hysteresis threshold
    low = 75
    high = 130
    hyst = apply_hysteresis_threshold(meije*255, low, high)

    return hyst

def apply_background_removal_sato(img):
    no_small = apply_background_removal_filtered(img)

    sat = sato(no_small, sigmas=range(1, 7, 1), black_ridges=False)
    # not bad with those parameters
    # the unnecessary bits are grayers around the main nerves
    # Histeresis could be good for that
    # Apply hysteresis threshold
    low = 75
    high = 100
    lowt = (sat*255 > low).astype(int)
    hight = (sat*255 > high).astype(int)
    hyst = apply_hysteresis_threshold(sat*255, low, high)

    return hyst

def apply_background_removal_black_tophat(img):
    nrows, ncols = img.shape
    row, col = np.ogrid[:nrows, :ncols]
    invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
    equalized_image = apply_background_removal(img)

    # Attempt with black_tophat
    footprint = disk(1)
    black_tophat_image = black_tophat(equalized_image, footprint)
    black = util.invert(equalized_image+black_tophat_image)
    black[invalid_pixels] = 0

    # Might actually be good enough to perform hysterisis
    low = 150
    high = 200
    hyst = apply_hysteresis_threshold(black*255, low, high)

    # Little white dots to remove
    no_small = remove_small_objects(hyst, 75) # empirical value 

    return no_small