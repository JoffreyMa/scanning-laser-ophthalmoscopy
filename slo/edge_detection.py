import cv2
from os import path as osp

input_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/input/images_IOSTAR'
output_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/output'
image_path = osp.join(input_dir, 'star01_OSC.jpg')

gaussian_blur_path = osp.join(output_dir, 'star01_OSC_blur_7x7_sd2.jpg')

canny_7_path = osp.join(output_dir, 'star01_OSC_canny_7.jpg')

# Read the original image
img = cv2.imread(image_path)
 
# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Remove noise
#img_gray=cv2.fastNlMeansDenoising(img_gray,None,h=10,templateWindowSize=7,searchWindowSize=7)

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, ksize=(7,7), sigmaX=2) 
cv2.imwrite(gaussian_blur_path, img_blur)
 
# Canny Edge Detection
edges_7 = cv2.Canny(image=img_blur, threshold1=0, threshold2=8000, apertureSize=7) # Canny Edge Detection
# Save Canny Edge Detection Image
cv2.imwrite(canny_7_path, edges_7)
