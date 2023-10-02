import os

# config values 

Config = {"table_detection_padding_pixel":10, 'table_recognition_padding_pixel':5,
        'table_detection_threshold':0.7, 'table_recognition_threshold':0.8,
        'table_padd': 20, 'row_padd':6, 'cell_padd':3,
        }
        
tesseract_config = {'tesseractpath':'C://Program Files//Tesseract-OCR//tesseract.exe'}

model_config = {'detection_model_path':'D:/Table-detection/models/detection-model',
                'recognition_model_path':'D:/Table-detection/models/recognition-model'}