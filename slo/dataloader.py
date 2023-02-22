# Somewhat like a dataloader, function returning an iterable over the images and their label
import os
import os.path as osp
from PIL import Image

def dataload(image_dir='data/input/images_IOSTAR', label_dir='data/input/label'):
    image_names = os.listdir(image_dir)
    label_names = os.listdir(label_dir)
    # Assume the pattern starXX_OSC.jpg for the image and GT_XX.png for the label 
    res = [{'image_name':image_name, 
    'label_name':label_name, 
    'image':Image.open(osp.join(image_dir, image_name)), 
    'label':Image.open(osp.join(label_dir, label_name))} for image_name in image_names for label_name in label_names if image_name[4:6]==label_name[3:5]]
    return res

if __name__ == '__main__':
    print(dataload())