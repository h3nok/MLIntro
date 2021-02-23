
import io

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from core.image.features import TensorFlowFeature as tff
from extension.tf_extension import ImageReader
from preprocessing.image_augmenter import ImageAugmenter
import cv2 as cv


class FrameDatum:

    def __init__(self, datum=None, frame_data=None, frame_id=None, groundtruth=None):
        """

        Abstract datatype representing a  2D image and two of its attributes.
        Expects an input of datum of type dataframe with columns frame_data, frame_id
        and category

        @param frame_id:
        @param frame_data:
        @param groundtruth:
        """
        assert datum is not None or (frame_id, frame_data, groundtruth)

        if frame_data:
            if isinstance(frame_data, pd.Series):
                frame_data = frame_data[0]
            else:
                frame_data = frame_data
        else:
            assert not datum.empty
            assert isinstance(datum, pd.DataFrame)
            frame_data = datum['frame_data'][0]
            frame_id = datum['frame_id'][0]
            groundtruth = datum['category'][0]

        self.__bytes = bytes(frame_data)
        self.__groundtruth = groundtruth
        self.__frame_id = frame_id
        self.__array = None
        self.__augmenter = None

        self.__augment_function = None

    @property
    def numpy_array(self):
        return self.__array

    @property
    def shape(self):
        if not self.__array:
            self.to_numpy_array()
        return self.__array.shape

    @property
    def dims(self):
        height, width = ImageReader.read_image_dims(self.__bytes)
        return height, width

    @property
    def groundtruth(self):
        return self.__groundtruth

    @property
    def datum(self):
        return self.__bytes

    @property
    def frame_id(self):
        return self.__frame_id

    @property
    def cvmat(self):
        if not self.__array:
            self.to_numpy_array()
        return cv.imdecode(self.__array, cv.IMREAD_COLOR)

    def augment(self, augment_cmd=None):
        """

        @param augment_cmd:
        @return:
        """
        self.__augmenter = ImageAugmenter(self.__bytes, frame_id=self.__frame_id)
        self.__augment_function = {
            'x9': self.__augmenter.augment_x9(),
            'x6': self.__augmenter.augment_x6(),
            'x5': self.__augmenter.augment_x5(),
            'x3': self.__augmenter.augment_x3(),
            'x2': self.__augmenter.augment_x2(),
            'x1': self.__augmenter.augment_x1()
        }

        if augment_cmd not in self.__augment_function.keys():
            raise ValueError("Unknown augment command. Please supply available "
                             "command from the following {}".format(self.__augment_function.keys()))

        augmented_frames = []

        for frame_id, frame_data in self.__augment_function[augment_cmd].items():
            augmented_frames.append(FrameDatum(frame_data=frame_data,
                                               frame_id=frame_id,
                                               groundtruth=self.groundtruth))
        return augmented_frames

    def to_numpy_array(self):
        """
        Converts raw bytes to numpy array. Easier datatype to manipulate
        @return:
        """
        # self.__array = np.array(Image.open(io.BytesIO(self.__bytes)), np.uint8)[..., :3]
        self.__array = np.fromstring(self.__bytes, np.uint8).reshape(1, -1)

    def to_tfexample(self, label_code, image_format=b'png'):
        """

        @param image_format: typically b'png' - raw bytes
        @param label_code: integer representation of the frame label
        @return:
        """
        try:
            height, width = self.dims

            return tf.train.Example(features=tf.train.Features(feature={
                'image/encoded': tff.bytes_feature(self.__bytes),
                'image/format': tff.bytes_feature(image_format),
                'image/class/label': tff.int64_feature(label_code),
                'image/height': tff.int64_feature(height),
                'image/width': tff.int64_feature(width),
                'image/frame-id': tff.bytes_feature(self.__frame_id.encode())
            }))

        except BaseException as e:
            print(e)
            return None

    def show(self):
        im = Image.fromarray(self.__array)
        b = io.BytesIO()
        im.save(b, 'png')
        im.show()
        del b
