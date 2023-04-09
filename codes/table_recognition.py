import os

from datatypes.datatypes import TableRecognitionData, TableDetectionData
from codes.image_processing import ImageProcessor
from datatypes.config import Config

class TableRecognition:
    def __init__(self, feature_extractor, recognition_model, threshold):
        self.feature_extractor = feature_extractor
        self.recognition_model = recognition_model
        self.threshold = threshold

    def table_recognition_from_detection(self, recognition_image, detection_results):

        for table in detection_results.tables:
            recognised_table_results = TableRecognitionData()
            bbox = table.detection_box
            detected_tbl = recognition_image.crop(bbox)
            img_processor = ImageProcessor()
            padded_table = img_processor.image_padding(image=detected_tbl, padd=Config['table_padd'])
            width, height = padded_table.size

            recognition_encoding = self.feature_extractor(padded_table, return_tensors='pt')
            recognition_output = self.recognition_model(**recognition_encoding)
            recognition_results = self.feature_extractor.post_process_object_detection(recognition_output, threshold=0.7, target_sizes=[(height, width)])
            recognition_results = recognition_results[0]

            recognised_table_results.scores = (recognition_results['scores'].tolist())
            recognised_table_results.labels = (recognition_results['labels'].tolist())
            recognised_table_results.boxes = (recognition_results['boxes'].tolist())

            table.recognitiondata.append(recognised_table_results)
        return detection_results