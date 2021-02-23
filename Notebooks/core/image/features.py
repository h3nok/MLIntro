import tensorflow as tf


class TensorFlowFeature:

    @staticmethod
    def bytes_feature(values):
        """Returns a TF-Feature of bytes.

        Args:
        values: A string.

        Returns:
            A TF-Feature.
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

    @staticmethod
    def float_feature(values):
        """Returns a TF-Feature of floats.

        Args:
          values: A scalar of list of values.

        Returns:
          A TF-Feature.
        """
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    @staticmethod
    def int64_feature(values):
        """Returns a TF-Feature of int64s.

        Args:
          values: A scalar or list of values.

        Returns:
          A TF-Feature.
        """
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
