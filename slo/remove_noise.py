import cv2
from os import path as osp

input_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/input/images_IOSTAR'
output_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/output'
image_path = osp.join(input_dir, 'star01_OSC.jpg')

denoised1_path = osp.join(output_dir, 'star01_OSC_denoised1.jpg')

# Read the original image
img = cv2.imread(image_path)
# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

denoised1=cv2.fastNlMeansDenoising(img_gray,None,h=10,templateWindowSize=7,searchWindowSize=7)

# Save the denoised image
cv2.imwrite(denoised1_path, denoised1)