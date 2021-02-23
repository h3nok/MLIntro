from enum import Enum


class ModelArch(Enum):
    """
    Supported models
    """

    EfficientNet = {
        'viNet Version': '3.0',
        'Description': 'Supports Efficient net versions B0 - B7'
     }

    Inception = {
        'viNet Version': '2.0',
        'Description': "Various Inception models, "
                       "Version 2 is the base architecture for viNet 2.x"
    }

    OCR = {
        'viNet Version': "4.0",
        'Description': "Optical Character Recognition"
    }

