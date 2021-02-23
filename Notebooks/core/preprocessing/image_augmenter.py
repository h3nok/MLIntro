import io

import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
from PIL import Image


class ImageAugmenter(object):
    """
    Examples:
        # There are three options to augment frames
        # Option 1
        # add augmentation to sequence then call augment
        aug.blur(20)
        aug.resize(224, 224)
        aug_frame = aug.augment()

        # Option 2
        # add augmentation to sequence and call augment in one step
        # Other augmentations can be called before or after the blur
        aug.blur(20)
        aug_frame = aug.resize(224, 224, ret=True)

        # Option 3
        # Multi Augment method. pass in the desired augmentation methods and parameters
        methods = [aug.blur, aug.resize]
        params = [[20], [224, 224]]
        aug_frame = aug.multi_augmentation(methods, params)

    Currently ImageAugmenter has support for tf.Tensor np.ndarray and bytes data types

    For any difficulties getting imgaug refer to this link:
    https://gis.stackexchange.com/questions/62925/why-is-shapely-not-installing-correctly
    """
    __frame = None
    _random = True
    _return_type = None
    _augment_seq = None
    _frame_id = None

    def __init__(self, x, randomize=True, ret_type=bytes, frame_id=None):
        """
        :param x: Image to augment as bytes, np.arrays and tf.Tensor
        :param randomize: Set to true for random blurs which provided values being upper bounds
        """
        assert x is not None

        if ret_type is None:
            self._return_type = type(x)
        else:
            self._return_type = ret_type

        if isinstance(x, tf.Tensor):
            self.__frame = x.numpy()[..., :3]  # Convert to np.array
        elif isinstance(x, np.ndarray):
            self.__frame = x[..., :3]
        elif isinstance(x, bytes):
            self.__frame = np.array(Image.open(io.BytesIO(x)))[..., :3]
        self._random = randomize
        self._frame_id = frame_id
        self._augment_seq = iaa.Sequential([])

    def augment(self):
        """
        Once the imgaug sequential has been set this will return the augmented image. self._augment_seq cannot be None
        Once run this will clear the augmenting sequence
        :return:
        returns either tf.Tensor or np.ndarray or bytes. depending on what was inputted or what was specified in init
        """
        aug_frame = self._augment_seq.augment_image(self.__frame)
        self._augment_seq = iaa.Sequential([])

        if self._return_type == tf.Tensor:
            return tf.convert_to_tensor(aug_frame)
        elif self._return_type == np.ndarray:
            return aug_frame
        elif self._return_type == bytes:
            im = Image.fromarray(aug_frame)
            b = io.BytesIO()
            im.save(b, 'png')
            return bytes(b.getvalue())
        else:
            return None

    def multi_augmentation(self, augmentation_list, augmentation_param_list):
        """
        refer to example in class description
        :param augmentation_list: list of augmentation function from the desired class
        :param augmentation_param_list: list of list of params in the same order as appears on function
        """
        for i in range(len(augmentation_list)):
            if isinstance(augmentation_param_list[i], list):
                augmentation_list[i](*augmentation_param_list[i], ret=False)
            elif isinstance(augmentation_param_list[i], dict):
                augmentation_list[i](**augmentation_param_list[i], ret=False)
            elif augmentation_param_list is None:
                augmentation_list[i](ret=False)
        return self.augment()

    def blur(self, k=3, ret=False):
        """
        Blurs images, values for k > 10 recommended for heavy blur
        :param k: averaging kernal with size k
        :param ret: return augmented image
        """
        lower_k = k
        if self._random:
            lower_k = 0
        self._augment_seq.append(iaa.AverageBlur(k=(lower_k, k)))
        if ret:
            return self.augment()

    def motion_blur(self, k=4, angle=90, ret=False):
        """
        motion blur. works best when the angle is in line with the birds motion. k > 10 work well for heavy blur
        """
        lower_k = k
        lower_angle = angle
        if self._random:
            lower_k = 3
            lower_angle = 0
        self._augment_seq.append(iaa.MotionBlur(k=(lower_k, k), angle=(lower_angle, angle)))

        if ret:
            return self.augment()

    def rotate_shear(self, degrees=90, shear=0, mode='edge', ret=False):
        """

        @param degrees:
        @param shear:
        @param mode:
        @param ret:
        @return:
        """
        lower_degrees = degrees
        lower_shear = shear
        if self._random:
            lower_degrees = 0
            lower_shear = 0
        if degrees % 90 == 0 and shear == 0 and not self._random:
            k = int(degrees / 90)
            self._augment_seq.append(iaa.Rot90(k=k))
        else:
            self._augment_seq.append(iaa.Affine(rotate=(lower_degrees, degrees),
                                                shear=(lower_shear, shear), mode=mode))

        if ret:
            return self.augment()

    def flip_up_down(self, ret=False):
        """

        @param ret:
        @return:
        """
        prop = 1
        if self._random:
            prop = .5
        self._augment_seq.append(iaa.Flipud(prop))

        if ret:
            return self.augment()

    def flip_left_right(self, ret=False):
        """

        @param ret:
        @return:
        """
        prop = 1
        if self._random:
            prop = .5
        self._augment_seq.append(iaa.Fliplr(prop))

        if ret:
            return self.augment()

    def crop_from_center(self, fraction=.95, ret=False):
        """

        @param fraction:
        @param ret:
        @return:
        """
        fraction = 1 - fraction
        self._augment_seq.append(iaa.Crop(percent=fraction))

        if ret:
            return self.augment()

    def weather(self, weather_type='fog', ret=False):
        """
        Returns image with desired weather added to it
        :param weather_type: which weather to pick, see below for options
        :param ret: return augmented image or not
        """

        if weather_type == 'fog':
            self._augment_seq.append(iaa.Fog(deterministic=False))
        elif weather_type == 'light clouds':
            self._augment_seq.append(iaa.CloudLayer(
                (210, 230), (-2.5, -1.5), 5, 0, (0.45, .8), 1, (-3.0, -1.5), 1.1, (.7, 1.5)))
        elif weather_type == 'clouds':
            self._augment_seq.append(iaa.Clouds(deterministic=True))
        elif weather_type == 'snow':
            self._augment_seq.append(iaa.Snowflakes(deterministic=True))
        elif weather_type == 'snowylandscape':
            self._augment_seq.append(iaa.FastSnowyLandscape(deterministic=True))
        if ret:
            return self.augment()

    def adjust_brightness(self, v=20, ret=False):
        lower_v = v
        if self._random:
            lower_v = -1 * v
        self._augment_seq.append(iaa.Add(value=(lower_v, v)))
        if ret:
            return self.augment()

    def adjust_hue_saturation(self, v=15, ret=False):
        lower_v = v
        if self._random:
            lower_v = -1 * v
        self._augment_seq.append(iaa.AddToHueAndSaturation(value=(lower_v, v)))
        if ret:
            return self.augment()

    def adjust_contrast(self, gamma=1.12, ret=False):
        lower_gamma = gamma
        if self._random:
            lower_gamma = .5
        self._augment_seq.append(iaa.GammaContrast(gamma=(lower_gamma, gamma)))
        if ret:
            return self.augment()

    def perspective(self, scale=.1, ret=False):
        lower_scale = scale
        if self._random:
            lower_scale = 0
        self._augment_seq.append(iaa.PerspectiveTransform(scale=(lower_scale, scale)))
        if ret:
            return self.augment()

    def resize(self, h=224, w=224, ret=False):
        size = {'height': h, 'width': w}
        self._augment_seq.append(iaa.Resize(size=size))
        if ret:
            return self.augment()

    def noop(self, ret=False):
        self._augment_seq.append(iaa.Noop())
        if ret:
            return self.augment()

    def augment_x9(self):
        frames = [
            self.crop_from_center(.95, ret=True),
            self.adjust_brightness(15, ret=True),
            self.adjust_hue_saturation(15, True),
            self.adjust_contrast(1.5, True),
            self.perspective(.12, ret=True),
            self.rotate_shear(degrees=180, ret=True),
            self.rotate_shear(degrees=90, ret=True),
            self.rotate_shear(degrees=270, ret=True),
            self.multi_augmentation([self.flip_up_down,
                                     self.weather,
                                     self.adjust_contrast,
                                     self.adjust_hue_saturation],
                                    [None, dict(weather_type='light clouds'), dict(gamma=1.2), dict(v=15)])]
        frame_ids = [
            self._frame_id + '_crop_from_center.95',
            self._frame_id + '_adjust_brightness_15',
            self._frame_id + '_adjust_hue_saturation_15',
            self._frame_id + '_adjust_contrast_1dot5',
            self._frame_id + '_perspective_dot12',
            self._frame_id + '_rotated_180',
            self._frame_id + '_rotated_90',
            self._frame_id + '_rotated_270',
            self._frame_id + '_multi_fud_weather_contrast_hue_n_sat_light_cloud']

        return {frame_ids[i]: frames[i] for i in range(len(frame_ids))}

    def augment_x3(self):
        frames = [
            self.rotate_shear(degrees=180, ret=True),
            self.rotate_shear(degrees=90, ret=True),
            self.rotate_shear(degrees=270, ret=True)]
        frame_ids = [
            self._frame_id + '_rotated_180',
            self._frame_id + '_rotated_90',
            self._frame_id + '_rotated_270']

        return {frame_ids[i]: frames[i] for i in range(len(frame_ids))}

    def augment_x5(self):
        frames = [
            self.perspective(.14, ret=True),
            self.perspective(.16, ret=True),
            self.rotate_shear(degrees=180, ret=True),
            self.rotate_shear(degrees=90, ret=True),
            self.rotate_shear(degrees=270, ret=True)]
        frame_ids = [
            self._frame_id + '_perspective_dot14',
            self._frame_id + '_perspective_dot16',
            self._frame_id + '_rotated_180',
            self._frame_id + '_rotated_90',
            self._frame_id + '_rotated_270']

        return {frame_ids[i]: frames[i] for i in range(len(frame_ids))}

    def augment_x6(self):
        frames = [
            self.adjust_hue_saturation(15, True),
            self.perspective(.14, ret=True),
            self.perspective(.16, ret=True),
            self.rotate_shear(degrees=180, ret=True),
            self.rotate_shear(degrees=90, ret=True),
            self.rotate_shear(degrees=270, ret=True)]

        frame_ids = [
            self._frame_id + 'adjust_hu_sat_15',
            self._frame_id + '_perspective_dot14',
            self._frame_id + '_perspective_dot16',
            self._frame_id + '_rotated_180',
            self._frame_id + '_rotated_90',
            self._frame_id + '_rotated_270']

        return {frame_ids[i]: frames[i] for i in range(len(frame_ids))}

    def augment_x2(self):
        frames = [
            self.perspective(.14, ret=True),
            self.rotate_shear(degrees=180, ret=True),
            ]
        frame_ids = [
            self._frame_id + '_perspective_dot14',
            self._frame_id + '_rotated_180']

        return {frame_ids[i]: frames[i] for i in range(len(frame_ids))}

    def augment_x1(self):
        frames = [
            self.rotate_shear(degrees=180, ret=True),
        ]
        frame_ids = [
            self._frame_id + '_rotated_180']

        return {frame_ids[i]: frames[i] for i in range(len(frame_ids))}
