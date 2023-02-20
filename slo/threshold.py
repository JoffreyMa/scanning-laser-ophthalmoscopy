import cv2
from os import path as osp

input_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/input/images_IOSTAR'
output_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/output'
image_path = osp.join(input_dir, 'star01_OSC.jpg')

thresh_tozero_path = osp.join(output_dir, 'star01_OSC_thresh_tozero.jpg')

# Read the original image
img = cv2.imread(image_path)

ret,thresh = cv2.threshold(img,90,255,cv2.THRESH_TOZERO_INV)

# Save the threshold image
cv2.imwrite(thresh_tozero_path, thresh)