import tensorflow as tf

V1_FEATURES = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/class/label': tf.io.FixedLenFeature([], tf.int64),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64)
}

V2_FEATURES = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/class/label': tf.io.FixedLenFeature([], tf.int64),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/frame-id': tf.io.FixedLenFeature([], tf.string)
}


class TFRecordDecoder:
    def __init__(self, filepath):
        self.__filepath = filepath

    @staticmethod
    def _parse(file):
        example = tf.io.parse_example(file, V2_FEATURES)
        shape = (example['image/width'], example['image/height'], 3)
        label = example['image/class/label']
        frame_id = example['image/frame-id']
        frame = tf.image.decode_png(example['image/encoded'])

        return [frame, shape, label, frame_id]

    def decode_v2(self) -> list:
        """Extract features from a TFExample version 2

        :return: list of features
        """

        dataset = tf.data.TFRecordDataset(self.__filepath)
        dataset = dataset.map(self._parse)

        return dataset

    @staticmethod
    def decode_label(label: int, labels: dict):
        return labels[label]
