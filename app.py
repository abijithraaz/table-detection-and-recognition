import os

import time
from fastapi import  FastAPI, UploadFile, File
import uvicorn
from PIL import Image

from transformers import TableTransformerForObjectDetection, DetrFeatureExtractor
from codes.table_recognition import TableRecognition
from codes.table_detection import TableDetection
from codes.table_preprocessing import TablePreprocessor
from codes.data_extraction import TextDataExtraction

from datatypes.config import Config, tesseract_config

# Fast API the object creation
app = FastAPI()

# Table detection model loading
try:
    detection_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
except:
    print('Table detection model loading is failed!!')

# Recognition model loading
try:
    recognition_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
except:
    print('Table recognition model loading is failed!!')

# Detection feature extractor
detection_feature_extractor = DetrFeatureExtractor(do_resize=True, size=800, max_size=800)
# Recognition feature extractor 
recognition_feature_extractor = DetrFeatureExtractor(do_resize=True, size=1000, max_size=1000)

# config values for the detection and recognition

# Detection Object 
detection_obj = TableDetection(detection_feature_extractor, detection_model, threshold=Config['table_detection_threshold'])

# Recognition Object
recognition_obj = TableRecognition(recognition_feature_extractor, recognition_model, threshold=Config['table_recognition_threshold'])

table_preprocessor = TablePreprocessor()
textdataextractor = TextDataExtraction(tesseract_path=tesseract_config['tesseractpath'])


@app.get("/health")
def healthcheck():
    return "200"

@app.post('/table-data-extraction')
def table_data_extraction_from_image(file: UploadFile = File(...)):
    if not (file.filename.split('.')[-1]).lower() in ("jpg", "jpeg", "png"):
        return {'Image must be jpg or png format!'}
    print(f'#---------- Table extractor started {time.strftime("%Y-%m-%d %H:%M:%S")} -----------#')
    image = Image.open(file.file).convert('RGB')
    detection_result = detection_obj.table_detection_from_image(image)
    recognition_result = recognition_obj.table_recognition_from_detection(image, detection_result)
    preprocessed_tables = table_preprocessor.table_structure_sorting(recognition_result)
    exracted_table_data = textdataextractor.cell_data_extraction(image, preprocessed_tables)
    print(f'#---------- Table extractor ended {time.strftime("%Y-%m-%d %H:%M:%S")} -----------#\n')
    return exracted_table_data


# def table_data_extraction_from_image1(file):
#     image = Image.open(file).convert('RGB')
#     detection_result = detection_obj.table_detection_from_image(image)
#     recognition_result = recognition_obj.table_recognition_from_detection(image, detection_result)
#     preprocessed_tables = table_preprocessor.table_structure_sorting(recognition_result)
#     exracted_table_data = textdataextractor.cell_data_extraction(image, preprocessed_tables)
#     return exracted_table_data

# if __name__ == '__main__':
#     # extraction = table_data_extraction_from_image1('D:\\Table-detection\\example-images\\2tables.png')
#     # print(extraction)
#     uvicorn.run("app:app", port=5000, reload=False, access_log=False)
