import os
import io
import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
from core.preprocessing.image_augmenter import ImageAugmenter


TEST_IMAGE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\\image\\test_images\\test_image1.png"


class TestImageAugmenter:
    def test_bytes(self):
        image = np.array(Image.open(TEST_IMAGE_PATH))
        aug = ImageAugmenter(image, False, ret_type=bytes)
        # Validation that bytes works
        assert Image.fromarray(np.array(Image.open(TEST_IMAGE_PATH))[..., :3]) == Image.fromarray(
            np.array(Image.open(io.BytesIO(aug.noop(ret=True))))[..., :3])

    def test_augment(self):
        image = np.array(Image.open(TEST_IMAGE_PATH))[..., :3]
        aug = ImageAugmenter(image, False, ret_type=np.ndarray)
        assert aug.crop_from_center(.95, ret=True) is not None

    def test_multi_augmentation(self):
        image = np.array(Image.open(TEST_IMAGE_PATH))[..., :3]
        aug = ImageAugmenter(image, False, ret_type=np.ndarray)
        assert aug.multi_augmentation([aug.flip_up_down, aug.weather, aug.adjust_contrast,
                                       aug.adjust_hue_saturation], [None, ['light clouds'], [1.2], [15]]) is not None

    def test_augment_save(self):
        image_path = r'C:\Users\ijorquera\Desktop\Defect Images\images\Color\Bad Mend Sample 1 Color-Bad_Mend_Sample_1_Color-1_000004.bmp'
        extension = os.path.splitext(image_path)[1]
        path = os.path.dirname(os.path.realpath(__file__))

        image = np.array(Image.open(image_path))[..., :3]

        aug = ImageAugmenter(image, False, ret_type=np.ndarray)
        augmented_dict = aug.augment_x9()

        for key in augmented_dict.keys():
            augmented_dict[key].save(f"{path}key{extension}")
