from skimage.filters import threshold_multiotsu, sato, apply_hysteresis_threshold
import numpy as np
from skimage import graph, morphology
import skimage.io
from os import path as osp

input_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/input/images_IOSTAR'
output_dir = '/home/users/jma-21/IA716 - Perception pour les systèmes autonomes/scanning-laser-ophthalmoscopy/data/output'
image_path = osp.join(input_dir, 'star01_OSC.jpg')

otsu_path = osp.join(output_dir, 'star01_OSC_otsu.jpg')
sato_path = osp.join(output_dir, 'star01_OSC_sato.jpg')
sato_thres_path = osp.join(output_dir, 'star01_OSC_sato_thres.jpg')

lowt_path = osp.join(output_dir, 'star01_OSC_sato_lowt.jpg')
hight_path = osp.join(output_dir, 'star01_OSC_sato_hight.jpg')

skeleton_path = osp.join(output_dir, 'star01_OSC_skeleton.jpg')

no_disk_path = osp.join(output_dir, 'star01_OSC_sato_thres_nodisk.jpg')

# Read the image
image = skimage.io.imread(image_path, as_gray=True)

# Avoid black corners with Otsu
t0, t1 = threshold_multiotsu(image, classes=3)
mask = (image > t0)

# Save the otsu image
skimage.io.imsave(otsu_path, mask)

# Sato detects vessels, there seems to be about 3 or 4 levels of vessels
vessels = sato(image, sigmas=[3]) * mask

# Save the sato image
skimage.io.imsave(sato_path, vessels)

# Apply histeresis threshold
low = 3
high = 8

lowt = (vessels > low).astype(int)
hight = (vessels > high).astype(int)
hyst = apply_hysteresis_threshold(vessels, low, high)

# Save the histeresis images
skimage.io.imsave(lowt_path, lowt)
skimage.io.imsave(hight_path, hight)
skimage.io.imsave(sato_thres_path, hyst)

# Offset disk issue
# There is always a disk 
# To remove it find center of the graph and remove around it
skeleton = morphology.skeletonize(hyst)

skimage.io.imsave(skeleton_path, skeleton)


g, nodes = graph.pixel_graph(skeleton, connectivity=2)
px, distances = graph.central_pixel(
        g, nodes=nodes, shape=skeleton.shape, partition_size=100
        )

print(px)

radius = 30 # estimated based on visual interpretation of images
rr, cc = skimage.draw.disk((px[1], px[0]), radius, shape=image.shape)

# Draw the black disk on the image by setting the corresponding pixel values to 0
hyst[rr, cc] = 0

# Save the histeresis images without the disk
skimage.io.imsave(no_disk_path, hyst)

# Too bad the center is off