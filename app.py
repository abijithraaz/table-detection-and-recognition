import os
import platform
import time
import logging
from fastapi import  FastAPI, UploadFile, File
import uvicorn
import pytesseract
import streamlit as st
import pandas as pd
from PIL import Image
from typing import List

from transformers import TableTransformerForObjectDetection, DetrFeatureExtractor
from codes.table_recognition import TableRecognition
from codes.table_detection import TableDetection
from codes.table_preprocessing import TablePreprocessor
from codes.data_extraction import TextDataExtraction
from datatypes.config import Config, tesseract_config, model_config

if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = tesseract_config['tesseractpath']

# Table detection-recognition model loading function
@st.cache_resource
def load_models():
    try:
        # models loading from local
        # detection_model = TableTransformerForObjectDetection.from_pretrained(model_config['detection_model_path'])
        # recognition_model = TableTransformerForObjectDetection.from_pretrained(model_config['recognition_model_path'])

        # models loading from hugginfacehub
        detection_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        recognition_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")

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

# # Fast API the service if we need to install this as a microservice
# app = FastAPI()

# @app.get("/health")
# def healthcheck():
#     return "200"

# @app.post('/table-data-extraction')
# def table_data_extraction_from_image(file: UploadFile = File(...)):
#     if not (file.filename.split('.')[-1]).lower() in ("jpg", "jpeg", "png"):
#         return {'Image must be jpg or png format!'}
#     print(f'#---------- Table extractor started {time.strftime("%Y-%m-%d %H:%M:%S")} -----------#')
#     image = Image.open(file.file).convert('RGB')
#     detection_result = detection_obj.table_detection_from_image(image)
#     recognition_result = recognition_obj.table_recognition_from_detection(image, detection_result)
#     preprocessed_tables = table_preprocessor.table_structure_sorting(recognition_result)
#     exracted_table_data = textdataextractor.cell_data_extraction(image, preprocessed_tables)
#     print(f'#---------- Table extractor ended {time.strftime("%Y-%m-%d %H:%M:%S")} -----------#\n')
#     return exracted_table_data

def convert_to_df(extracted_object):
    logging.info(f'#---------- Table visualization started {time.strftime("%Y-%m-%d %H:%M:%S")} -----------#')
    def _show_outputdf(table_list:List[List], table_number:int):
        op_df = pd.DataFrame(table_list)
        container.write(f'Extracted tabel: {table_number}')
        container.dataframe(op_df)
        container.write('\n')
    if len(extracted_object.tables) != 0:
        table_no = 1
        for table in extracted_object.tables:
            table_list = []
            for row in table.extracted_rows:
                row_list = []
                for cell in row.extracted_cells:
                    row_list.append(cell.value)
                table_list.append(row_list)
            _show_outputdf(table_list=table_list, table_number=table_no)
            table_no += 1
    else:
        container.write('No tables are predicted!!!!')

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
if __name__ == '__main__':
    st.title('Table detection and recognition')
    st.write('Table data extraction application with help of microsoft detr models.')
    image = st.sidebar.file_uploader(label='Upload image file for data extraction', type=['png','jpg','jpeg','tif'])
    if image:
        result = st.sidebar.button(label='Predict', on_click=table_data_extraction_from_image1, args=(image,))
        container = st.container()
        container.subheader('Extracted tables :snowflake:')