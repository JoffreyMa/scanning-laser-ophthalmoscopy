import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk

from os import path as osp

input_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/input/images_IOSTAR'
output_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/output'
image_path = osp.join(input_dir, 'star01_OSC.jpg')

entropy_path = osp.join(output_dir, 'star01_OSC_entropy.jpg')

# Read the original image
img = cv2.imread(image_path)
# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_entropy = entropy(img_gray, footprint=disk(1000), out=None, mask=None, shift_x=False, shift_y=False, shift_z=False)

cv2.imwrite(entropy_path, img_entropy)