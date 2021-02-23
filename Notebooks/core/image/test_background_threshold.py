import os
from unittest import TestCase
import image.background_threshold as bt
import numpy as np
from cv2 import imshow, waitKey, imread, absdiff

TEST_IMAGE_DIR = "test_images"
TRUTH_IMAGE_DIR = "test_images\\truth"
TEST_IMAGE1_PATH = "test_image1.png"
TEST_IMAGE2_PATH = "test_image2.png"
TEST_IMAGE3_PATH = "test_image3.png"
TEST_IMAGE4_PATH = "test_image4.png"
TEST_IMAGE5_PATH = "test_image5.png"
TRUTH_IMAGE1_PATH = "test_image1_truth.png"
TRUTH_IMAGE2_PATH = "test_image2_truth.png"
TRUTH_IMAGE3_PATH = "test_image3_truth.png"
TRUTH_IMAGE4_PATH = "test_image4_truth.png"
TRUTH_IMAGE5_PATH = "test_image5_truth.png"
TRUTH_IMAGE5A_PATH = "test_image7_truth.png"
TRUTH_IMAGE5B_PATH = "test_image8_truth.png"

RNG_SEED = 0

# I don't know what the best way to test it
TURN_ON_VISUAL_TESTS = False


def image_eq(image_1_path, image_2_path):
    img1 = imread(image_1_path)
    img2 = imread(image_2_path)

    return np.all(img1 == img2)


class TestBackgroundThresholder(TestCase):
    def test_grab_foreground1(self):
        # Just test to make sure the whole process does not fail
        # setup file paths
        img_path = os.path.join(TEST_IMAGE_DIR, TEST_IMAGE1_PATH)
        truth_img_path = os.path.join(TRUTH_IMAGE_DIR, TRUTH_IMAGE1_PATH)
        image_dst_path = os.path.join(TRUTH_IMAGE_DIR, "tmp.png")

        # init the BackgroundThresholder class
        bg_thr = bt.BackgroundThresholder.from_image_path(img_path)
        bg_thr.set_rng_seed(RNG_SEED)
        bg_thr.set_bounding_box(11, 9, 101, 68)  # I found these numbers using ms paint

        # call BackgroundThresholder::grab_foreground
        _mat = bg_thr.grab_foreground(None)  # return image
        bg_thr.grab_foreground(image_dst_path)  # save to file

        # assert
        assert image_eq(image_dst_path, truth_img_path)

        # clean up
        os.remove(image_dst_path)

    def test_grab_foreground2_ave_color(self):
        img_path = os.path.join(TEST_IMAGE_DIR, TEST_IMAGE2_PATH)
        truth_img_path = os.path.join(TRUTH_IMAGE_DIR, TRUTH_IMAGE2_PATH)
        image_dst_path = os.path.join(TRUTH_IMAGE_DIR, "tmp.png")

        bg_thr = bt.BackgroundThresholder.from_image_path(img_path)
        bg_thr.set_rng_seed(RNG_SEED)
        bg_thr.set_bounding_box(14, 10, 80-14, 89-10)  # I found these numbers using ms paint

        bg_thr.grab_foreground(image_dst_path, (-1, -1, -1))  # ave color

        assert image_eq(image_dst_path, truth_img_path)

        # clean up
        os.remove(image_dst_path)

    def test_grab_foreground3_remove_spots(self):
        img_path = os.path.join(TEST_IMAGE_DIR, TEST_IMAGE3_PATH)
        truth_img_path = os.path.join(TRUTH_IMAGE_DIR, TRUTH_IMAGE3_PATH)
        image_dst_path = os.path.join(TRUTH_IMAGE_DIR, "tmp.png")

        bg_thr = bt.BackgroundThresholder.from_image_path(img_path)
        bg_thr.set_rng_seed(RNG_SEED)
        bg_thr.set_bounding_box(8, 10, 149-8, 168-10)  # I found these numbers using ms paint

        # there should not be spots/holes in the foreground object
        if TURN_ON_VISUAL_TESTS:
            mat = bg_thr.grab_foreground(None, (0, 255, 0))  # green
            imshow("visual test", mat)
            waitKey(0)
            assert input("Does the image have holes in the foreground object? (Y/N): ").lower()[0] == 'n'
        else:
            bg_thr.grab_foreground(image_dst_path, (0, 255, 0))  # green

            assert image_eq(image_dst_path, truth_img_path)

            # clean up
            os.remove(image_dst_path)

    def test_grab_foreground4_bad_bounding_box(self):
        img_path = os.path.join(TEST_IMAGE_DIR, TEST_IMAGE4_PATH)
        truth_img_path = os.path.join(TRUTH_IMAGE_DIR, TRUTH_IMAGE4_PATH)
        image_dst_path = os.path.join(TRUTH_IMAGE_DIR, "tmp.png")

        bg_thr = bt.BackgroundThresholder.from_image_path(img_path)
        bg_thr.set_rng_seed(RNG_SEED)

        # I found these numbers using ms paint, they are intentionally wrong. Should be (7, 7, 316-7, 107-7).
        # This will cause the bird's right wing to be cut off a little
        bg_thr.set_bounding_box(41, 7, 316-41, 107-7)

        if TURN_ON_VISUAL_TESTS:
            mat = bg_thr.grab_foreground(None, (0, 255, 0))
            imshow("visual test", mat)
            waitKey(0)
            assert input("Was the foreground object detected ok? (Y/N): ").lower()[0] == 'y'
        else:
            bg_thr.grab_foreground(image_dst_path, (0, 255, 0))  # green

            assert image_eq(image_dst_path, truth_img_path)

            # clean up
            os.remove(image_dst_path)

    def test_grab_foreground5_small_bounding_box(self):
        img_path = os.path.join(TEST_IMAGE_DIR, TEST_IMAGE5_PATH)
        truth_img_path = os.path.join(TRUTH_IMAGE_DIR, TRUTH_IMAGE5_PATH)
        image_dst_path = os.path.join(TRUTH_IMAGE_DIR, "tmp.png")

        bg_thr = bt.BackgroundThresholder.from_image_path(img_path)
        bg_thr.set_rng_seed(RNG_SEED)
        # image.shape = (178, 141)
        bg_thr.set_bounding_box(1, 1, 178-2, 141-2)  # smallest bounding box

        if TURN_ON_VISUAL_TESTS:
            mat = bg_thr.grab_foreground(None, (0, 255, 0))
            imshow("visual test", mat)
            waitKey(0)
            assert input("Was the foreground object detected? (Y/N): ").lower()[0] == 'y'
        else:
            bg_thr.grab_foreground(image_dst_path, (0, 255, 0))  # green

            assert image_eq(image_dst_path, truth_img_path)

            # clean up
            os.remove(image_dst_path)

    def test_grab_foreground6_twice_in_a_row(self):
        # Just test to make sure the whole process does not fail
        img_path = os.path.join(TEST_IMAGE_DIR, TEST_IMAGE1_PATH)
        truth_img_path = os.path.join(TRUTH_IMAGE_DIR, TRUTH_IMAGE1_PATH)
        image_dst_path1 = os.path.join(TRUTH_IMAGE_DIR, "tmp1.png")
        image_dst_path2 = os.path.join(TRUTH_IMAGE_DIR, "tmp2.png")

        bg_thr1 = bt.BackgroundThresholder.from_image_path(img_path)
        bg_thr1.set_rng_seed(RNG_SEED)
        bg_thr1.set_bounding_box(11, 9, 101, 68)  # I found these numbers using ms paint

        bg_thr1.grab_foreground(image_dst_path1)  # save to file

        bg_thr2 = bt.BackgroundThresholder.from_image_path(img_path)
        bg_thr2.set_rng_seed(RNG_SEED)
        bg_thr2.set_bounding_box(11, 9, 101, 68)  # I found these numbers using ms paint

        bg_thr2.grab_foreground(image_dst_path2)  # save to file

        assert image_eq(image_dst_path1, image_dst_path2)
        assert image_eq(image_dst_path1, truth_img_path)
        assert image_eq(image_dst_path2, truth_img_path)

        # clean up
        os.remove(image_dst_path1)
        os.remove(image_dst_path2)

    def test_grab_foreground7_alpha(self):
        img_path = os.path.join(TEST_IMAGE_DIR, TEST_IMAGE5_PATH)
        truth_img_path = os.path.join(TRUTH_IMAGE_DIR, TRUTH_IMAGE5A_PATH)
        image_dst_path = os.path.join(TRUTH_IMAGE_DIR, "tmp.png")

        bg_thr = bt.BackgroundThresholder.from_image_path(img_path)
        bg_thr.set_rng_seed(RNG_SEED)
        bg_thr.set_bounding_box(7, 9, 178-7, 141-9)  # I found these numbers using ms paint

        bg_thr.grab_foreground(image_dst_path, (0, 0, 0, 0))  # add alpha

        assert image_eq(image_dst_path, truth_img_path)

        # clean up
        os.remove(image_dst_path)

    def test_grab_foreground8_outline_mode(self):
        img_path = os.path.join(TEST_IMAGE_DIR, TEST_IMAGE5_PATH)
        truth_img_path = os.path.join(TRUTH_IMAGE_DIR, TRUTH_IMAGE5B_PATH)
        image_dst_path = os.path.join(TRUTH_IMAGE_DIR, "tmp.png")

        bg_thr = bt.BackgroundThresholder.from_image_path(img_path)
        bg_thr.set_rng_seed(RNG_SEED)
        bg_thr.set_outline_mode()
        bg_thr.set_bounding_box(7, 9, 178-7, 141-9)  # I found these numbers using ms paint

        if TURN_ON_VISUAL_TESTS:
            mat = bg_thr.grab_foreground(None, (0, 255, 0))
            imshow("visual test", mat)
            waitKey(0)
            assert input("Was the foreground object detected? (Y/N): ").lower()[0] == 'y'
        else:
            bg_thr.grab_foreground(image_dst_path, (0, 255, 0))  # green

            assert image_eq(image_dst_path, truth_img_path)

            # clean up
            os.remove(image_dst_path)
