import os

from datatypes.datatypes import DetectionLabels, TableRecognitionOrdered


class TablePreprocessor():
    def table_structure_sorting(self, table_data):
        for table in table_data.tables:
            recognized_row = []
            recognized_column = []
            recognized_ord_obj = TableRecognitionOrdered([])
            # print(table.recognitiondata[0])
            for score, label, box in zip(table.recognitiondata[0].scores, table.recognitiondata[0].labels, table.recognitiondata[0].boxes):
                # print(score, label, box)
                newbox = []
                if label == DetectionLabels.table_row.value:
                    newbox = box
                    newbox.append(label)
                    newbox.append(score)
                    recognized_row.append(newbox)
                if label == DetectionLabels.table_column.value:
                    newbox = box
                    newbox.append(label)
                    newbox.append(score)
                    recognized_column.append(newbox)

            recognized_row.sort(key=lambda x:x[1])
            recognized_column.sort(key=lambda x:x[0])

            recognized_ord_obj.recognized_row = recognized_row
            recognized_ord_obj.recognized_column = recognized_column
            table.ordered_recognitiondata.append(recognized_ord_obj)

        return table_data