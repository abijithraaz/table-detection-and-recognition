import os
import re
import pytesseract
from pytesseract import Output
from datatypes.datatypes import Row, Cell
from codes.image_processing import ImageProcessor
from datatypes.config import Config

class TextDataExtraction():
    def __init__(self):
        pass
    
    def clean_ocr_data(self, value):
        transf = ''.join(e for e in value if e==' 'or e=='.' or e.isalnum())
        transf.strip()
        return transf
        
    def pytess(self, cell_pil_img):
        return ' '.join(pytesseract.image_to_data(cell_pil_img, output_type=Output.DICT, config='-c tessedit_char_blacklist=œ˜â€œï¬â™Ã©œ¢!|”?«“¥ --psm 6 preserve_interword_spaces')['text']).strip()

    def cell_data_extraction(self, image,  table_data):
        for table in table_data.tables:
            tableimg_processor = ImageProcessor()
            table_bbox = table.detection_box
            table_image = image.crop(table_bbox)
            table_image = tableimg_processor.image_padding(table_image, padd=Config['table_padd'])

            for row_idx, table_row in enumerate(table.ordered_recognitiondata[0].recognized_row):
                row_obj = Row([])
                xmin_row, ymin_row, xmax_row, ymax_row, _, _ = table_row

                row_image = table_image.crop((xmin_row,ymin_row,xmax_row,ymax_row))
                row_width, row_height = row_image.size
                row_obj.rowindex = row_idx

                # Cell bounding box creation
                xa, ya, xb, yb = 0, 0, 0, row_height

                for indx, table_column in enumerate(table.ordered_recognitiondata[0].recognized_column):
                    cell_obj = Cell()
                    xmin_col, _, xmax_col, _,_,_ = table_column
                    xmin_col, xmax_col = xmin_col -Config['table_padd'], xmax_col - Config['table_padd']
                    xa = xmin_col
                    xb = xmax_col
                    if indx == 0:
                        xa = 0
                    if indx == len(table.ordered_recognitiondata[0].recognized_column)-1:
                        xb = row_width
                    
                    cell_img = row_image.crop((xa, ya, xb, yb))
                    xa, ya, xb, yb = xa, ya, xb, yb

                    cell_value = self.pytess(cell_img)
                    transformed_cell_value = self.clean_ocr_data(cell_value)

                    cell_obj.cellindex = indx
                    cell_obj.value = transformed_cell_value

                    row_obj.extracted_cells.append(cell_obj)
                table.extracted_rows.append(row_obj)
                    
        return table_data
