import numpy as np
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.filters.rank import entropy, enhance_contrast_percentile
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
import math
from skimage import data, filters
from matplotlib import pyplot as plt
from dataloader import dataload

def my_segmentation(img, img_mask, seuil):
    img_out = (img_mask & (img < seuil))
    return img_out

def evaluate(img_out, img_GT):
    GT_skel  = thin(img_GT, max_iter = 15) # On suppose que la demie epaisseur maximum 
    img_out_skel  = thin(img_out, max_iter = 15) # d'un vaisseau est de 15 pixels...
    TP = np.sum(img_out_skel & img_GT) # Vrais positifs
    FP = np.sum(img_out_skel & ~img_GT) # Faux positifs
    FN = np.sum(GT_skel & ~img_out) # Faux negatifs

    ACCU = TP / (TP + FP) # Precision
    RECALL = TP / (TP + FN) # Rappel
    return ACCU, RECALL, img_out_skel, GT_skel

def evaluate_process(process=my_segmentation, verbose=True, **kwargs):
    data = dataload()
    for i, d in enumerate(data):
        # Open image with grayscale
        img = np.asarray(d['image']).astype(np.uint8)
        nrows, ncols = img.shape
        row, col = np.ogrid[:nrows, :ncols]
        # Consider pixels in inner disk only
        img_mask = (np.ones(img.shape)).astype(np.bool_)
        invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
        img_mask[invalid_pixels] = 0
        # Open ground truth as bool
        img_GT =  np.asarray(d['label']).astype(np.bool_)

        # Apply method
        img_out = process(img, img_mask, **kwargs)

        # Compute metrics
        ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
        data[i]['accuracy'] = ACCU
        data[i]['recall'] = RECALL
        if verbose:
            print(f"Image:{d['image_name']} | Accuracy={np.round(ACCU, 5)} | Recall={np.round(RECALL, 5)}")
    return data

if __name__ == '__main__':
    evaluate_process(seuil=100)