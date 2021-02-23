import efficientnet.tfkeras as efn
from tensorflow.keras.applications import (DenseNet121, DenseNet169, InceptionResNetV2, InceptionV3, MobileNet,
                                           MobileNetV2, NASNetMobile, Xception)

ClassificationNet = {
    # EfficientNet
    'B0': efn.EfficientNetB0,
    'B1': efn.EfficientNetB1,
    'B2': efn.EfficientNetB2,
    'B3': efn.EfficientNetB3,
    'B4': efn.EfficientNetB4,
    'B5': efn.EfficientNetB5,
    'B6': efn.EfficientNetB6,
    'B7': efn.EfficientNetB7,
    'L2': efn.EfficientNetL2,

    # Other Models - Small in size, ideal for embedded devices
    'InceptionV3': InceptionV3,
    'InceptionResNet': InceptionResNetV2,
    'Xception': Xception,
    'MobileNet': MobileNet,
    'MobileNetV2': MobileNetV2,
    'DenseNet121': DenseNet121,
    'DenseNet169': DenseNet169,
    'NASNetMobile': NASNetMobile
}
