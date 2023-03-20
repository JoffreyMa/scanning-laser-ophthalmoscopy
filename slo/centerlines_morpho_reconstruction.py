import numpy as np
from skimage import io, color, filters, morphology, measure, feature
from scipy import ndimage as ndi
from os import path as osp
import skimage.io

input_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/input/images_IOSTAR'
output_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/output'
image_path = osp.join(input_dir, 'star21_OSC.jpg')

segmented_vessels_normalized_path = osp.join(output_dir, 'star21_OSC_segmented_vessels_normalized.jpg')
segmented_vessels_preprocessing_path = osp.join(output_dir, 'star21_OSC_segmented_vessels_preprocessing.jpg')
segmented_vessels_centerline_path = osp.join(output_dir, 'star21_OSC_segmented_vessels_centerline.jpg')
segmented_vessels_path = osp.join(output_dir, 'star21_OSC_segmented_vessels.jpg')

def background_normalization(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    background = filters.rank.mean(img, footprint=kernel)
    normalized_img = img - background
    return normalized_img

def normalize_image(img):
    img_min, img_max = np.min(img), np.max(img)
    return (img - img_min) / (img_max - img_min)

def thin_vessel_enhancement(normalized_img, orientations):
    max_response = np.zeros_like(normalized_img)
    for angle in orientations:
        gabor_real, gabor_imag = filters.gabor(normalized_img, frequency=2.47, theta=angle)
        response = np.abs(gabor_real)
        max_response = np.maximum(max_response, response)
    enhanced_img = normalized_img + max_response
    return normalize_image(enhanced_img)

def vessel_centerline_detection(enhanced_img, orientations):
    dog_filters = [filters.gabor_kernel(1, theta=angle, sigma_x=1, sigma_y=1, offset=0.5) for angle in orientations]
    vessel_candidates = np.stack([ndi.convolve(enhanced_img, filt) for filt in dog_filters])
    candidates = np.argmax(vessel_candidates, axis=0)
    centerlines = np.zeros_like(enhanced_img, dtype=bool)
    for i, filt in enumerate(dog_filters):
        centerlines = np.logical_or(centerlines, candidates == i)
    centerlines = morphology.remove_small_objects(centerlines, min_size=30)
    return centerlines

def vessel_segmentation(centerlines):
    enhanced_vessels = filters.frangi(centerlines)
    binary_vessels = np.zeros_like(enhanced_vessels, dtype=bool)
    for threshold in np.linspace(0.1, 1, 4):
        binary_vessels = np.logical_or(binary_vessels, enhanced_vessels > threshold)
    filled_vessels = morphology.binary_dilation(binary_vessels)
    return filled_vessels

def vessel_detection_pipeline(img_path):
    # Read and convert image to grayscale
    img_gray = skimage.io.imread(img_path, as_gray=True)

    # Preprocessing
    normalized_img = background_normalization(img_gray, kernel_size=15)
    skimage.io.imsave(segmented_vessels_normalized_path, normalized_img)
    enhanced_img = thin_vessel_enhancement(normalized_img, orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4])
    skimage.io.imsave(segmented_vessels_preprocessing_path, enhanced_img)

    # Vessel centerline detection
    centerlines = vessel_centerline_detection(enhanced_img, orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4])
    skimage.io.imsave(segmented_vessels_centerline_path, centerlines)

    # Vessel segmentation
    segmented_vessels = vessel_segmentation(centerlines)
    skimage.io.imsave(segmented_vessels_path, segmented_vessels)

    return segmented_vessels

# Run the pipeline
segmented_vessels = vessel_detection_pipeline(image_path)

