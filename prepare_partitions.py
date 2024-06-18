import os
import json
import glob
import numpy as np
import cv2 as cv
from pycocotools import mask
from skimage.measure import label
import shutil

def main():

    data_root_dir = 'data/BUS-UCLM'
    output_dir = 'data/partitions'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'masks'), exist_ok=True)

    test_cases = ['COPE', 'ANFO', 'ELCO', 'CRCI', 'FLKA']
    images = [x for x in glob.glob(os.path.join(data_root_dir,'images/*.png'))]


    for img_path in images:
        case = os.path.basename(img_path)[:4]
        image = os.path.basename(img_path)
        if case in test_cases:
            shutil.copy(os.path.join(data_root_dir, 'images', image), os.path.join(output_dir, 'test', 'images', image))
            shutil.copy(os.path.join(data_root_dir, 'masks', image), os.path.join(output_dir, 'test', 'masks', image))
        else:
            shutil.copy(os.path.join(data_root_dir, 'images', image), os.path.join(output_dir, 'train', 'images', image))
            shutil.copy(os.path.join(data_root_dir, 'masks', image), os.path.join(output_dir, 'train', 'masks', image))
    

if __name__ == "__main__":
    main()