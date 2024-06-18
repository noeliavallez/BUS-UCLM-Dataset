import os
import json
import glob
import numpy as np
import cv2 as cv
from pycocotools import mask
from skimage.measure import label


def create_coco_dict():
    return  {   
                "licenses": [
                    {
                        "name": "",
                        "id": 0,
                        "url": ""
                    }
                ],
                "info": {
                    "contributor": "",
                    "date_created": "",
                    "description": "",
                    "url": "",
                    "version": "",
                    "year": ""
                },
                "categories": [
                    {
                        "id": 1,
                        "name": "Benign",
                        "supercategory": ""
                    },
                    {
                        "id": 2,
                        "name": "Malignant",
                        "supercategory": ""
                    }
                ],
                "images": [],
                "annotations": []
            }


def create_image_info(image_id, file_name, image_size, date_captured=0, license_id=0, coco_url="", flickr_url=""):

    return {
                "id": image_id,
                "width": image_size[0],
                "height": image_size[1],
                "file_name": file_name,
                "license": license_id,
                "coco_url": coco_url,
                "flickr_url": flickr_url,
                "date_captured": date_captured
            }


def create_annotation_info(annotation_id, image_id, class_id, binary_mask):

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    bounding_box = mask.toBbox(binary_mask_encoded)
    area = mask.area(binary_mask_encoded)
    segmentation = region_contour(binary_mask)

    return {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "segmentation": [segmentation],
                "area": area.tolist(),
                "bbox": bounding_box.tolist(),
                "iscrowd": 0,
                "attributes": {
                    "occluded": False,
                    "rotation": 0
                }
            }


def region_contour(binary_mask):
    contours = cv.findContours(binary_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour = []
    for p in contours[0][0]:
        contour.append(int(p.flatten()[0]))
        contour.append(int(p.flatten()[1]))
    return contour


def read_img_and_mask(IMG_PATH, MASK_PATH):

    image = cv.imread(IMG_PATH)
    mask = cv.imread(MASK_PATH)
    
    b,g,r = cv.split(mask)
    mask = 1*(g==255) + 2*(r==255)
    mask = mask.astype(np.uint8)
    
    return image, mask


def main():

    data_root_dir = 'data/BUS-UCLM'
    output_dir = os.path.join('data')

    train_set = create_coco_dict()
    val_set = create_coco_dict()
    test_set = create_coco_dict()
    all_set = create_coco_dict()

    val_cases = []#['MENE', 'COVA', 'SIBA', 'FLBA', 'PAGY']
    test_cases = ['COPE', 'ANFO', 'ELCO', 'CRCI', 'FLKA']
    images = [x for x in glob.glob(os.path.join(data_root_dir,'images/*.png'))]

    train_images = {'coco':train_set, 'set_name':'train', 'img_paths':[]}
    val_images = {'coco':val_set, 'set_name':'val', 'img_paths':[]}
    test_images = {'coco':test_set, 'set_name':'test', 'img_paths':[]}
    all_images = {'coco':all_set, 'set_name':'all', 'img_paths':[]}

    for img_path in images:
        case = os.path.basename(img_path)[:4]
        if case in val_cases:
            val_images['img_paths'].append(img_path)
        elif case in test_cases:
            test_images['img_paths'].append(img_path)
        else:
            train_images['img_paths'].append(img_path)
        all_images['img_paths'].append(img_path)
    
    for image_set in [all_images, train_images, val_images, test_images]:

        img_id = 1
        label_id = 1

        for img_path in image_set['img_paths']:
            
            msk_path = img_path.replace('images','masks')
            img, img_mask = read_img_and_mask(img_path, msk_path)

            image_info = create_image_info(img_id,img_path.replace(os.path.join(data_root_dir,'images')+'/',''),img.shape[:2])
            image_set['coco']['images'].append(image_info)

            for class_id in [1, 2]:
                if np.any(img_mask==class_id):
                    aux_mask = (img_mask==class_id).astype(np.uint8)
                    regions = label(aux_mask)
                    reg = 1
                    while np.any(regions==reg):
                        annotation_info = create_annotation_info(label_id, img_id, class_id, (regions==reg).astype(np.uint8))
                        image_set['coco']['annotations'].append(annotation_info)
                        label_id += 1
                        reg += 1
            img_id += 1

        if image_set['img_paths']:
            with open(os.path.join(output_dir, image_set['set_name']+'.json'), 'w') as f:
                json.dump(image_set['coco'], f, indent=4)


if __name__ == "__main__":
    main()