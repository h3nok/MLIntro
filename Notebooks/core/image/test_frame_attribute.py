from core.image.frame_attribute import FrameAttribute
from unittest import TestCase

# Image path on my local computer
TEST_IMAGE_PATH = r'test_images/test_image2.png'

class TestFrameAttribute(TestCase):
    fa = FrameAttribute(TEST_IMAGE_PATH)

    def test_perceived_brightness(self):
        # Checks to make sure we don't have None
        assert self.fa.perceived_brightness() is not None

    def test_brightness(self):
        assert self.fa.brightness() is not None
        assert self.fa.brightness(mode='rms') is not None

    def test_variance(self):
        assert self.fa.variance() is not None

    def test_sharpness(self):
        val = self.fa.sharpness()
        assert val is not None

    def test_entropy(self):
        assert self.fa.entropy() is not None

    def test_noise(self):
        assert self.fa.noise() is not None

    def test_contrast(self):
        assert self.fa.contrast() is not None

    def test_fourier_spectrum_iq(self):
        assert self.fa.fourier_spectrum_iq() is not None

    def test_dominant_colors(self):
        num_colors = 5
        colors = self.fa.dominant_colors(amount=num_colors)
        assert len(colors) == num_colors

        # a color could be in the format (r,g,b) or (r,g,b,a)
        assert all([len(c) == 3 or len(c) == 4 for c in colors])

        # check to make sure we can never break the function
        assert(self.fa.dominant_colors(amount=100000000))

    def test_average_color(self):
        assert len(self.fa.average_color()) == 3

    def test_centroid(self):
        assert len(self.fa.centroid()) == 2

    def test_attribute_to_single_val(self):
        assert self.fa.attribute_to_single_val('dominant_colors', fa_option='blue')
