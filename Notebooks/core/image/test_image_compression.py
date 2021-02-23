from unittest import TestCase
import os
from image.image_compression import Compressor

# These images are committed to the repository. I looked at a lot of other image related
# repositories and they handled their tests by committing the test images to the repository.
# they are both < 20kb so it should be fine.
# I got them from viNet `source.frame_data`.
TEST_IMAGE1_PATH = "test_images/test_image1.png"
TEST_IMAGE2_PATH = "test_images/test_image2.png"


class TestCompression(TestCase):
    def test_compress_to_jpg_with_quality(self):
        comp = Compressor.from_image_path(TEST_IMAGE1_PATH)
        dst_path = comp.compress_to_jpg(10)

        assert os.path.splitext(dst_path)[1].lower() == '.jpg'  # we don't care what the whole path is
        assert os.path.exists(dst_path)

        # clean up
        os.remove(dst_path)

    def test_encode_to_jpg(self):
        comp = Compressor.from_image_path(TEST_IMAGE2_PATH)
        encoding = comp.encode_to_jpg(100)

        assert encoding is not None  # I don't really know what to do here
        assert len(encoding) > 0

    def test_encode_to_png(self):
        comp = Compressor.from_image_path(TEST_IMAGE1_PATH)
        encoding = comp.encode_to_png(9)

        assert encoding is not None  # I don't really know what to do here
        assert len(encoding) > 0

    def test_compress_to_jpg_with_dst(self):
        image_path = TEST_IMAGE1_PATH
        comp = Compressor.from_image_path(image_path)
        image_dst_path = os.path.splitext(image_path)[0] + '_out.jpg'
        dst_path = comp.compress_to_jpg(95, image_dst_path)

        assert dst_path == image_dst_path
        assert os.path.exists(dst_path)

        # clean up
        os.remove(dst_path)

    def test_compress_png(self):
        image_path = TEST_IMAGE2_PATH
        comp = Compressor.from_image_path(image_path)

        image_dst_path = os.path.splitext(image_path)[0] + '_out.png'
        dst_path = comp.compress_png(9, image_dst_path)

        assert dst_path == image_dst_path
        assert os.path.exists(dst_path)

        # clean up
        os.remove(dst_path)

    def test_compression_ratio(self):
        image_path = TEST_IMAGE1_PATH
        comp = Compressor.from_image_path(image_path)
        dst_path_jpg = comp.compress_to_jpg(100)  # lowest compression option
        dst_path_png = comp.compress_png(9)  # Highest compression option

        assert os.path.splitext(dst_path_jpg)[1].lower() == '.jpg'
        assert os.path.exists(dst_path_jpg)

        assert os.path.splitext(dst_path_png)[1].lower() == '.png'
        assert os.path.exists(dst_path_png)

        src_size = os.path.getsize(image_path)
        dst_size_jpg = os.path.getsize(dst_path_jpg)
        dst_size_png = os.path.getsize(dst_path_png)

        pseudo_ratio_jpg = dst_size_jpg / src_size
        pseudo_ratio_png = dst_size_png / src_size

        print("JPG % new size:", pseudo_ratio_jpg * 100.0)
        print("PNG % new size:", pseudo_ratio_png * 100.0)

        # We want to make sure that the resulting image is smaller in file size
        assert pseudo_ratio_jpg < 1.0
        assert pseudo_ratio_png < 1.0

        # clean up
        os.remove(dst_path_jpg)
        os.remove(dst_path_png)

    def test_open_we(self):
        with open(TEST_IMAGE1_PATH, 'rb') as f:
            image_data = f.read()

        comp = Compressor.from_image_codex(image_data)
        dst_path = comp.compress_to_jpg(10)

        assert os.path.splitext(dst_path)[1].lower() == '.jpg'  # we don't care what the whole path is
        assert os.path.exists(dst_path)

        # clean up
        os.remove(dst_path)

