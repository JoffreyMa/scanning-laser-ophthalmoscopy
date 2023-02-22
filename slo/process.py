# File containing all the tested process for this project
# all process functions must take an uint8 grayscale image as input and return an img_out

import cv2

def detect_edges(img):
    img_denoised = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=7)
    img_blur = cv2.GaussianBlur(img_denoised, ksize=(7,7), sigmaX=2)
    img_out = cv2.Canny(image=img_blur, threshold1=0, threshold2=8000, apertureSize=7)
    return img_out