import tensorflow as tf


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    @staticmethod
    def read_image_dims(image_data):
        assert isinstance(image_data, bytes)
        # image = self.decode_png(sess, image_data)
        image = tf.io.decode_png(image_data)
        return image.shape[0], image.shape[1]

    @staticmethod
    def decode_jpeg(image_data):
        image = tf.io.decode_jpeg(image_data)
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    @staticmethod
    def decode_png(image_data):
        image = tf.io.decode_png(image_data)
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

