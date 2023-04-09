import os

from datatypes.datatypes import ImageData
from datatypes.datatypes import TableDetectionData

class TableDetection():
    def __init__(self, feature_extractor, detection_model, threshold):
        self.feature_extractor = feature_extractor
        self.detection_model = detection_model
        self.threshold = threshold

    def table_detection_from_image(self, detection_image):

        table_data_extraction = ImageData([])
        image_width, image_height = detection_image.size
        detection_encoding = self.feature_extractor(detection_image, return_tensors='pt')
        detection_output = self.detection_model(**detection_encoding)
        detection_results = self.feature_extractor.post_process_object_detection(detection_output, threshold=0.3, target_sizes=[(image_height, image_width)])
        detection_results = detection_results[0]
        # copying the detections
        for score, label, bbox in zip((detection_results['scores']).tolist(), (detection_results['labels']).tolist(), (detection_results['boxes']).tolist()):
            detection_table_results = TableDetectionData()
            detection_table_results.detection_score = score
            detection_table_results.detection_label = label
            detection_table_results.detection_box = bbox
            table_data_extraction.tables.append(detection_table_results)
        return table_data_extraction

