# BUS-UCLM Dataset

This repository contains instructions to download and manipulate the BUS-UCLM segmentation dataset of breast ultrasound images.

## Description

Instructions to download the dataset and scripts to convert the annotations to COCO json format and to superpose the masks on top of the image are provided.

## Getting Started

### Python Dependencies

* opencv-python==4.10.0.84
* scikit-image==0.23.2
* pycocotools==2.0.8

### Preparing the data

* Clone this repository
* Download the dataset
```
wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/7fvgj4jsp7-1.zip
```
* Unzip it
```
unzip 7fvgj4jsp7-1.zip
```
* Rename the folder and remove the zip file
```
mv "BUS-UCLM Breast ultrasound lesion segmentation dataset" "data"
rm 7fvgj4jsp7-1.zip
```

### Executing the scripts

* To extract the COCO JSON annotation:
```
python extract_coco_annotations.py
```
* To draw the segmentation mask on top of the images run the command bellow. It will create a new folder in with the generated images are stored.
```
python draw_annotations.py
```
* To divide the dataset into train and test sets:
```
python prepare_partitions.py
```
```
* To anonymize the DICOM files (including the text in the pixeldata):
```
python anonymize.py
```

## Authors

Noelia Vallez, Gloria Bueno, Oscar Deniz, Miguel Angel Rienda, and Carlos Pastor

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the CC-BY-4.0 License - see the LICENSE.md file for details

## References

### Dataset

Vallez, Noelia; Bueno, Gloria; Deniz, Oscar; Rienda, Miguel Angel; Pastor, Carlos (2024), “BUS-UCLM: Breast ultrasound lesion segmentation dataset”, Mendeley Data, V1, doi: 10.17632/7fvgj4jsp7.1

### Papers

