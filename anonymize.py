
import os
import shutil
import randomname
import pydicom
from pydicom.uid import ImplicitVRLittleEndian
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


# List of tags to anonymize
tags_to_anonymize = [
    (0x0010, 0x0010),  # Patient's Name
    (0x0010, 0x0020),  # Patient ID
    (0x0010, 0x0030),  # Patient's Birth Date
    (0x0010, 0x0040),  # Patient's Sex
    (0x0008, 0x0020),  # Study Date
    (0x0008, 0x0030),  # Study Time
    (0x0008, 0x0090),  # Referring Physician's Name
    (0x0008, 0x0050),  # Accession Number
    (0x0008, 0x0080),  # Institution Name
    (0x0008, 0x1010),  # Station Name
    (0x0008, 0x1030),  # Study Description
    (0x0008, 0x1040),  # Institutional Department Name
    (0x0008, 0x1048),  # Physician(s) of Record
    (0x0008, 0x1050),  # Performing Physician's Name
    (0x0008, 0x1060),  # Name of Physician(s) Reading Study
    (0x0008, 0x1070),  # Operator's Name
    (0x0008, 0x1080),  # Admitting Diagnoses Description
    (0x0010, 0x1000),  # Other Patient IDs
    (0x0010, 0x1001),  # Other Patient Names
    (0x0010, 0x1010),  # Patient's Age
    (0x0010, 0x1020),  # Patient's Size
    (0x0010, 0x1030),  # Patient's Weight
    (0x0010, 0x1090),  # Medical Record Locator
    (0x0020, 0x000D),  # Study Instance UID
    (0x0020, 0x000E),  # Series Instance UID
    (0x0020, 0x0010),  # Study ID
    (0x0020, 0x0052),  # Frame of Reference UID
    (0x0020, 0x0200),  # Synchronization Frame of Reference UID
    (0x0040, 0x0244),  # Performed Procedure Step Start Date
    (0x0040, 0x0245),  # Performed Procedure Step Start Time
    (0x0040, 0x0253),  # Performed Procedure Step ID
    (0x0040, 0x0254),  # Performed Procedure Step Description
    (0x0040, 0xA124),  # UID
    (0x0040, 0xA730),  # Content Sequence
    (0x0088, 0x0140),  # Storage Media File-set UID
    (0x3006, 0x0024),  # Referenced Frame of Reference UID
    (0x3006, 0x00C2),  # Related Frame of Reference UID
    (0x0008, 0x0012),  # Instance Creation Date
    (0x0008, 0x0013),  # Instance Creation Time
    (0x0008, 0x0018),  # SOP Instance UID
    (0x0008, 0x0021),  # Series Date
    (0x0008, 0x0022),  # Acquisition Date
    (0x0008, 0x0023),  # Content Date
    (0x0008, 0x0031),  # Series Time
    (0x0008, 0x0032),  # Acquisition Time
    (0x0008, 0x0033),  # Content Time
    (0x0008, 0x1110),  # Referenced Study Sequence
    (0x0008, 0x1111),  # Referenced Performed Procedure Step Sequence
    (0x0032, 0x1032),  # Requesting Physician
    (0x0010, 0x1040),  # Patient's Address
    (0x0010, 0x2154),  # Patient's Telephone Numbers
    (0x0010, 0x1060),  # Patient's Mother's Birth Name
    (0x0010, 0x0032),  # Patient's Birth Time
    (0x0010, 0x21F0),  # Patient's Religious Preference
    (0x0008, 0x0081),  # Institution Address
    (0x0008, 0x0092),  # Referring Physician's Address
    (0x0008, 0x0094),  # Referring Physician's Telephone Numbers
    (0x0018, 0x1030),  # Protocol Name
    (0x0018, 0x1000),  # Device Serial Number
    (0x0018, 0x1020),  # Software Versions
    (0x0008, 0x103E),  # Series Description
    (0x0002, 0x0003),  # Media Storage SOP Instance UID
]


def load_dicom_image(file_path):
    dicom_image = pydicom.dcmread(file_path, force=True)
    if not hasattr(dicom_image.file_meta, 'TransferSyntaxUID'):
        dicom_image.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    if not hasattr(dicom_image, 'PlanarConfiguration'):
        dicom_image.PlanarConfiguration = 0
    image_data = dicom_image.pixel_array
    image_data = image_data.astype(np.uint8)
    return dicom_image, image_data


def anonymize_text(image, yolo_model):
    if len(image.shape) == 2: 
        input_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        input_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        input_image = image
    else:
        raise ValueError("Formato de imagen no soportado")
    results = yolo_model(input_image)
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return image


def anonymize_pixel_data(file_path):
    try:
        dicom_image, image_data = load_dicom_image(file_path)
        # YOLO text detection from: https://github.com/Alfadhils/YOLOv8-CRNN-Scene-Text-Recognition
        yolo_model = YOLO('yolov8_5k.pt')
        anonymized_image = anonymize_text(image_data, yolo_model)
        #plt.imshow(anonymized_image, cmap='gray')
        #plt.title('DICOM Image without Text')
        #plt.axis('off')
        #plt.show()
        save_dicom_image(dicom_image, anonymized_image, file_path)
    except Exception as e:
        print(f"Error: {e}")


def save_dicom_image(dicom_image, modified_image, output_path):
    dicom_image.PixelData = modified_image.tobytes()
    dicom_image.save_as(output_path)


def anonymize_imgs(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith(".dcm"):
                file_path = os.path.join(dirpath, filename)
                anonymize_pixel_data(file_path)


def anonymize_folder_and_files(root_path):
    names = {}
    unique = []
    for fold_name in os.listdir(root_folder):
        new = randomname.get_name().upper()
        new = new.split('-')[0][:2] + new.split('-')[1][:2]
        while new in unique:
            new = randomname.get_name().upper()
            new = new.split('-')[0][:2] + new.split('-')[1][:2]
        unique.append(new)
        name = os.path.basename(fold_name).split('.')[0]
        names[name] = new
    counts = {}
    for name in names:
        counts[name] = 0
    for dirname in os.listdir(root_folder):
        dirpath = os.path.join(root_folder, dirname)
        name = os.path.basename(dirname).split('.')[0]
        newdirname = names[name]
        newdirpath = dirpath.replace(dirname, newdirname)
        shutil.move(dirpath, newdirpath)
        for filename in os.listdir(newdirpath):
            if filename.endswith(".dcm"):
                filepath = os.path.join(root_folder, newdirname, filename)
                name = filename.split('.')[0]
                newfilename = names[name] + '_' + str(counts[name]).zfill(3) + '.dcm'
                newfilepath = os.path.join(root_folder, newdirname, newfilename)
                shutil.move(filepath, newfilepath)
                counts[name] += 1
            else:
                # remove info data
                os.remove(os.path.join(root_folder, newdirname, filename))


def anonymize_dicom(root_path):
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith(".dcm"):
                file_path = os.path.join(dirpath, filename)
                dicom_data = pydicom.dcmread(file_path, force=True)

                # Function to anonymize a dataset
                def anonymize_dataset(dataset):
                    for tag in tags_to_anonymize:
                        if tag in dataset:
                            if isinstance(dataset[tag].value, pydicom.sequence.Sequence):
                                for item in dataset[tag].value:
                                    anonymize_dataset(item)
                            else:
                                vr = dataset[tag].VR
                                if vr == "PN":      # Name
                                    dataset[tag].value = 'Anonymized'
                                elif vr == "AS":    # Age
                                    dataset[tag].value = '000Y'
                                elif vr == "DA":    # Date
                                    dataset[tag].value = '19000101'
                                elif vr == "TM":    # Time
                                    dataset[tag].value = '000000.000000'
                                elif vr == "CS":    # Code String
                                    dataset[tag].value = 'ANON'
                                elif vr == "UI":    # UID
                                    dataset[tag].value = pydicom.uid.generate_uid()
                                else:
                                    dataset[tag].value = 'Anonymized'

                # Anonymize the main dataset
                anonymize_dataset(dicom_data)

                # Request Attributes Sequence
                if (0x0040, 0x0275) in dicom_data:
                    for item in dicom_data[0x0040, 0x0275]:
                        if 'ScheduledProcedureStepID' in item:
                            item.ScheduledProcedureStepID = 'Anonymized'
                        if 'RequestedProcedureID' in item:
                            item.RequestedProcedureID = 'Anonymized'

                dicom_data.save_as(file_path)



root_folder = 'path_dicom_root_folder'

RENAME_FOLDER_AND_FILES = False
ANNOMYMIZE_TAGS = False
ANNOMYMIZE_IMGS = True

if RENAME_FOLDER_AND_FILES:
    anonymize_folder_and_files(root_folder)
if ANNOMYMIZE_TAGS:
    anonymize_dicom(root_folder)
if ANNOMYMIZE_IMGS:
    anonymize_imgs(root_folder)

