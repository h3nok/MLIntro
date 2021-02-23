from PIL import Image, ImageStat
import numpy as np
import cv2
import skimage
import skimage.measure
import skimage.restoration
import scipy.fftpack as fp
import math
from matplotlib import colors


class FrameAttribute:
    _grayscale_frame_data = None
    _fft_freq_img = None

    def __init__(self, sample):
        """
        with provided sample will automatically generate both the np.array and the pil image. And
        height and width data
        @param sample: either in the form of a np.array or file location
        @return: None
        """
        if isinstance(sample, str):
            self._pil_frame_data = Image.open(sample)
            self._cv_frame_data = np.asarray(self._pil_frame_data.convert("RGB"))
        elif isinstance(sample, np.ndarray):
            self._pil_frame_data = Image.fromarray(sample)
            self._cv_frame_data = sample
        else:
            raise TypeError(f"{type(sample)} not an allowed type for 'sample'")

        self._width, self._height = self._cv_frame_data.shape[0], self._cv_frame_data.shape[1]

    def perceived_brightness(self) -> float:
        stat = ImageStat.Stat(self._pil_frame_data)
        r, g, b = stat.rms[0], stat.rms[1], stat.rms[2]
        return round(math.sqrt(0.241 * (r ** 2) +
                               0.691 * (g ** 2) +
                               0.068 * (b ** 2)))

    def brightness(self, mode='mean') -> float:
        """
        this calculated the average intensity, when mode = 'mean' other wise calculates
        the rms of the intensity
        @param mode: can specify between different methods of calculating the brightness
        @return: integer value for brightness
        """
        if self._grayscale_frame_data is None:
            self._grayscale_frame_data = self._pil_frame_data.convert('L')
        stat = ImageStat.Stat(self._grayscale_frame_data)
        if mode == 'mean':
            return round(stat.mean[0])
        else:
            return round(stat.rms[0])

    def variance(self) -> float:
        """
        This attribute helps determine how sharp an image is. or the Blurriness of the image
        lower the number the more blurry it may be.
        @return: Rounded value for variance
        """
        if self._grayscale_frame_data is None:
            self._grayscale_frame_data = self._pil_frame_data.convert('L')
        return round(cv2.Laplacian(np.array(self._grayscale_frame_data), cv2.CV_64F).var())

    def sharpness(self, offset=50.0, lookrange=7) -> float:
        """
        Finds the sharpness of the image.
        It uses two metrics: `max_diff` which is from the point in the image with the most change between pixels and
        `pixel_value_range` which is the contrast in the chunk around the max_diff pixel.
        Note, the higher the difference between the two metrics, the higher the blur (generally).

        This method works well when the background is relatively constant/blurry and we don't want that to effect the
        sharpness score. It falls short when there is high blur in one direction but the image looks very sharp in
        another direction.

        The idea for this algorithm is from this stack overflow post:
        https://stackoverflow.com/a/21127948/9664285

        :param offset: the higher the value the more `max_diff` is important else the more the difference between
            `max_diff` and `pixel_value_range` is important. `offset >= 0`.
        :param lookrange: the size of the chunk around the `max_diff` pixel.
        :return: 0-1 in most cases
        """

        # 0: Define Vars And Make Gray Scale Image
        image = cv2.cvtColor(self._cv_frame_data, cv2.COLOR_RGB2BGR)  # OpenCV does things differently
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edge = np.zeros(gray_image.shape, dtype=np.float)

        # 1: Apply Sobel Filter
        # note: max value for horz_edge and vert_edge are 255*4 = 1020
        horz_edge = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, None, 3, 1.0, 0.0, cv2.BORDER_REPLICATE)
        horz_edge = np.absolute(horz_edge)

        vert_edge = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, None, 3, 1.0, 0.0, cv2.BORDER_REPLICATE)
        vert_edge = np.absolute(vert_edge)

        # note: max value is also technically 255*4 but is very unlikely, more likely to be at 255*2 = 510
        # with the two sobels the highest value is 765
        cv2.addWeighted(vert_edge, 0.5, horz_edge, 0.5, 0, edge, cv2.CV_64F)

        # 2: Find Greatest Edge / Biggest Pixel Diff
        # note max value for `max_overall_diff_val` is about 255*2 = 510
        _min_overall_val, max_overall_diff_val, _min_loc, max_loc = cv2.minMaxLoc(edge)

        # 3: Find the Range of values in the area (the local contrast)
        x, y, w, h = max_loc[0] - lookrange // 2, max_loc[1] - lookrange // 2, lookrange, lookrange
        x, y, x2, y2 = max(x, 0), max(y, 0), min(x+w, gray_image.shape[0]), min(y+h, gray_image.shape[1])

        chunk = gray_image[y:y2, x:x2]
        min_chunk_val, max_chunk_val, _min_loc, _max_loc = cv2.minMaxLoc(chunk)
        pixel_value_range = max_chunk_val - min_chunk_val  # max value is 255

        # 4: Put it all together
        # the 2 and the 510 act to normalize the values
        return (max_overall_diff_val / (offset + 2.0 * pixel_value_range)) * (1.0 + offset / 510.0)

    def entropy(self) -> float:
        """
        This attribute tries to capture the ratio of relevant information to noise
        @return: Rounded value for entropy
        """
        return round(skimage.measure.shannon_entropy(self._pil_frame_data), 4)

    def centroid(self) -> float:
        """
        This returns the centroid. Position data on the bird. Im not sure how useful this might be
        """
        if self._grayscale_frame_data is None:
            self._grayscale_frame_data = self._pil_frame_data.convert('L')
        M = skimage.measure.moments(np.array(self._grayscale_frame_data))
        centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
        return centroid

    def noise(self) -> float:
        """
        This tells us how much noise is in the image. Higher the value the more noise there is.
        @return: Rounded value for noise
        """
        return round(skimage.restoration.estimate_sigma(self._cv_frame_data, multichannel=True,
                                                        average_sigmas=True), 4)

    def contrast(self) -> float:
        """
        Contrast is defined as the difference between the max and min intensities
        @return: Integer value for the contrast
        """
        if self._grayscale_frame_data is None:
            self._grayscale_frame_data = self._pil_frame_data.convert('L')
        return np.amax(self._grayscale_frame_data) - np.amin(self._grayscale_frame_data)

    def dominant_colors(self, amount=1) -> float:
        """
        returns most dominant colors.
        when amount > 1, it tends to find very similar colors due to noise. might be smart to do a color
        segmentation to group colors together but this will likely be slow.
        @param amount: specifics the number of dominant colors to return
        @return: array of dominant colors and number of elements equal to the param amount.
        """
        # Get colors from image object
        pixels = self._pil_frame_data.getcolors(self._width * self._height)
        # Sort them by count number(first element of tuple)
        sorted_pixels = sorted(pixels, key=lambda t: t[0], reverse=True)
        # Get the most frequent color
        amount = min(amount, len(sorted_pixels))
        dominant_colors = [color[1] for color in sorted_pixels[:amount]]
        return dominant_colors

    def average_color(self) -> float:
        """
        Returns the average color
        """
        single_pixel = self._pil_frame_data.resize((1, 1))

        return single_pixel.getpixel((0, 0))

    def fourier_spectrum_iq(self) -> float:
        """
        This attribute computes the overall quality of the image. The lower the number the clearer the picture
        may be.
        Input: Image I of size M×N.  Output: Image Quality measure (FM) where FM stands for
        Frequency Domain Image Blur
        Measure Step 1: Compute F which is the Fourier Transform representation of image I
        Step 2:  Find Fc which is obtained by shifting the origin of F to centre.
        Step 3: Calculate AF = abs (Fc) where AF is the absolute value of the centered Fourier
        transform of image I.
        Step 4:  Calculate M = max (AF) where M is the maximum value of the frequency component
        in F. Step
        5: Calculate TH = the total number of pixels in F whose pixel value > thres,
        where thres = M/1000.Step 6: Calculate Image Quality measure (FM) from equation (1).
        @return: rounded value
        """
        self._fft_freq_img = FFT(self._pil_frame_data).freq_image
        fft_centered = fp.fftshift(self._fft_freq_img)
        af = np.abs(fft_centered)
        m = np.max(af)
        _fft_img_quality_threshold = round(m / 1000)
        th = len(np.where(self._fft_freq_img > _fft_img_quality_threshold))

        return round((th / m * self._width), 4)

    def attribute_to_single_val(self, fa_func, fa_option=None):
        """
        Helps convert the not numeric results to numeric ones.
        for example with dominant colors it does dot product between a given color
        @param fa_func: function name to call. ex. 'fourier_spectrum_iq'
        @param fa_option: for functions with non numerical outputs, like dominant colors. this will make it numerical.
        ex. for dominant_colors and fa_option blue it will take the dot product between the result and blue.
        """
        fa_result = getattr(self, fa_func)()

        if fa_func == self.dominant_colors.__name__ or fa_func == self.average_color.__name__:
            if isinstance(fa_result, list):
                color_res = fa_result[0]
            else:
                color_res = fa_result
            color = [255 * c for c in colors.to_rgb(fa_option)]
            if len(color_res) > 3:
                color_res = color_res[:3]
            return np.dot(color, color_res) / (math.sqrt(np.dot(color, color))*math.sqrt(np.dot(color_res, color_res)))

        return fa_result


class FFT:
    """
        An interface to Fourier transform.
        Currently converts 2D topology data to frequency domain.
    """
    def __init__(self, data):
        self._frame_data = data
        self._convert_to_freq_domain = lambda freq: fp.rfft(fp.rfft(data, axis=0), axis=0)
        self._freq_img = self._convert_to_freq_domain(self._frame_data)

    @property
    def freq_image(self):
        return self._freq_img
