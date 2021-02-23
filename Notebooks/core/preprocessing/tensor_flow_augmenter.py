import math
import tensorflow as tf


class TFAugmenter(object):
    __frame = None

    def __init__(self, x: tf.Tensor):
        """
        Initialize object with frame data
        :param x:
        """
        self.__frame = x

    def rotate(self, degrees: int = 90) -> tf.Tensor:
        """
        Rotation augmentation
        :param degrees: degree in which to rotate
        """
        if degrees == 90:
            rotated = tf.image.rot90(self.__frame)
            return rotated

        rotated = tf.contrib.image.rotate(self.__frame, math.radians(degrees))
        return rotated

    def flip_up_down(self) -> tf.Tensor:
        """
        Flip augmentation
        Returns: Augmented image
        """
        return tf.image.flip_up_down(self.__frame)

    def flip_left_right(self) -> tf.Tensor:
        """
        Flip augmentation
        Returns: Augmented image
        """
        return tf.image.flip_left_right(self.__frame)

    def central_crop(self, fraction=0.7) -> tf.Tensor:
        """
        crop from the center
        :param fraction: amount of crop from center
        :return: cropped image
        """
        return tf.image.central_crop(self.__frame, central_fraction=fraction)

    def resize(self, size=(224, 224)):
        """
        resize image
        :param size: (new width, new height)
        :return: resized image
        """
        return tf.image.resize(self.__frame, size)

    def adjust_contrast(self, contrast_factor=0.7) -> tf.Tensor:
        """
        adjust contrast
        :param contrast_factor: amount of contrast shift
        :return: adjusted image
        """
        return tf.image.adjust_contrast(self.__frame, contrast_factor=contrast_factor)

    def adjust_hue(self, delta=0.4) -> tf.Tensor:
        """ adjust hue
        :param delta:
        :return:
        """
        return tf.image.adjust_hue(self.__frame, delta=delta)

    def adjust_saturation(self, factor=10):
        """ adjust saturation
        :param factor:
        :return:
        """
        return tf.image.adjust_saturation(self.__frame, factor)

    def adjust_brightness(self, delta=0.2) -> tf.Tensor:
        """adjust_brightness
        :param delta:
        :return:
        """
        return tf.image.adjust_brightness(self.__frame, delta=delta)

    def adjust_gamma(self, gamma=0.2) -> tf.Tensor:
        """adjust gamma, gamma > 1, brighter, gamma < 1 darker
        :param gamma:
        :return:
        """
        return tf.image.adjust_gamma(self.__frame, gamma=gamma)
