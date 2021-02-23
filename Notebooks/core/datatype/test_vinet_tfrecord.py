from unittest import TestCase
from tfrecord import TFRecordDecoder
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class TestTFRecord(TestCase):
    record = TFRecordDecoder(
        r"E:\viNet_RnD\Datasets\Vattenfall-Gotland\jpg_comp\tfrecords\train\vinet_train_00001-of-00001"
        r".tfrecord")
    record = record.decode_v2()
    labels = {0: 'Buzzard', 1: 'Eagle-Or-Kite', 2: 'Gull', 3: 'Other-Avian-Gotland'}
    for element in record:
        print(TFRecordDecoder.decode_label(int(element[2].numpy()), labels))
    pass
