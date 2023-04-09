import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict

class DetectionLabels(Enum):
    table = 0
    table_column = 1
    table_row = 2
    table_column_header = 3
    table_projected_row_header = 4
    table_spanning_cell = 5

class ExtractionContext(Enum):
    document = 1
    table = 2
    row = 3

@dataclass
class Cell:
    cellindex : int = 0
    value : str = ''
    prob : float = 0.5

@dataclass
class Row:
    rowindex : int = 0
    extracted_cells : List[Cell]= field(default_factory=lambda: [])

@dataclass
class TableRecognitionData:
    scores : List = field(default_factory=lambda: [])
    labels : List = field(default_factory=lambda: [])
    boxes : List = field(default_factory=lambda: [])

@dataclass
class TableRecognitionOrdered:
    recognized_row : List = field(default_factory=lambda: [])
    recognized_column : List = field(default_factory=lambda: [])

@dataclass
class TableDetectionData:
    detection_score : float = 0.0
    detection_label : int = 0
    detection_box : List = field(default_factory=lambda: [])
    recognitiondata : TableRecognitionData = field(default_factory=lambda: [])
    ordered_recognitiondata : List[TableRecognitionOrdered] = field(default_factory=lambda: [])
    extracted_rows : List[Row] = field(default_factory=lambda: [])


@dataclass
class ImageData:
    tables: List[TableDetectionData]
