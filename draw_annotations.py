from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import cv2 as cv
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


class myCOCO(COCO):

    def __init__(self, annotation_file=None):
        super().__init__(annotation_file)

    def showAnns(self, anns, draw_bbox=False):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in anns:
            if ann['category_id'] == 2:
                c = [0.5, 0, 0]
            elif ann['category_id'] == 1:
                c = [0, 0.5, 0]
            if 'segmentation' in ann:
                if type(ann['segmentation']) == list:
                    # polygon
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape((int(len(seg)/2), 2))
                        polygons.append(Polygon(poly))
                        color.append(c)
            if draw_bbox:
                [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
                np_poly = np.array(poly).reshape((4,2))
                polygons.append(Polygon(np_poly))
                color.append(c)
        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.3)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=1)
        ax.add_collection(p)


output_path = 'data/visualization'
os.makedirs(output_path, exist_ok=True)
dataDir  =  'data/BUS-UCLM/images'
annFiles = [
            #'data/train.json',
            #'data/test.json',
            'data/all.json'
            ]

for annFile in annFiles:

    coco=myCOCO(annFile)

    imgIds = coco.getImgIds()
    for iid in imgIds:
        img_info = coco.loadImgs(iid)[0]
        I = cv.imread(os.path.join(dataDir, img_info['file_name']))
        annIds = coco.getAnnIds(imgIds=img_info['id'])
        anns = coco.loadAnns(annIds)
        plt.imshow(I)
        plt.axis('off')
        coco.showAnns(anns, draw_bbox=False)
        #plt.show()
        plt.savefig(os.path.join(output_path, img_info['file_name']))
        plt.close()
