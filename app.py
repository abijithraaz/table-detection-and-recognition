import os
import platform
import time
import logging
from fastapi import  FastAPI, UploadFile, File
import uvicorn
import pytesseract
import streamlit as st
from PIL import Image

from transformers import TableTransformerForObjectDetection, DetrFeatureExtractor
from codes.table_recognition import TableRecognition
from codes.table_detection import TableDetection
from codes.table_preprocessing import TablePreprocessor
from codes.data_extraction import TextDataExtraction
from datatypes.config import Config, tesseract_config

if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = tesseract_config['tesseractpath']

# Table detection-recognition model loading function
@st.experimental_singleton
def load_models():
    try:
        # Detection model loading
        # detection_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        detection_model = TableTransformerForObjectDetection.from_pretrained("D:/Table-detection/models/detection-model")
        
        # Recognition model loading
        # recognition_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
        recognition_model = TableTransformerForObjectDetection.from_pretrained("D:/Table-detection/models/recognition-model")
        return detection_model, recognition_model
    except:
        print('Table detection or recognition model loading is failed!!')    

# Models loading
detection_model, recognition_model = load_models()

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
textdataextractor = TextDataExtraction()

'''
# Fast API the service if we need to install this as a microservice
app = FastAPI()

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
'''

def convert_to_df(extracted_tables):
    
    print('extracted_tables: ', extracted_tables)

def table_data_extraction_from_image1(file):
    logging.info(f'#---------- Table extractor started {time.strftime("%Y-%m-%d %H:%M:%S")} -----------#')
    image = Image.open(file).convert('RGB')
    detection_result = detection_obj.table_detection_from_image(image)
    recognition_result = recognition_obj.table_recognition_from_detection(image, detection_result)
    preprocessed_tables = table_preprocessor.table_structure_sorting(recognition_result)
    exracted_table_data = textdataextractor.cell_data_extraction(image, preprocessed_tables)
    convert_to_df(exracted_table_data)
    
    logging.info((f'#---------- Table extractor ended {time.strftime("%Y-%m-%d %H:%M:%S")} -----------#\n'))
    return exracted_table_data

st.title('Table detection and recognition')
st.write('Table data extraction application with help of microsoft detr models.')
image = st.sidebar.file_uploader(label='Upload image file for data extraction', type=['png','jpg','jpeg','tif'])
if image:
    result = st.sidebar.button(label='Predict', on_click=table_data_extraction_from_image1, args=(image,))
    st.text_input(label='Prediction', value=str(result))