import cv2
import numpy as np
import os


class Compressor:
    __image__ = None

    def __init__(self, image):
        self.__image__ = image
        self._src_path = None

    @classmethod
    def from_image_path(cls, image_path: str):
        """
        Initializes the Compressor class.

        :param image_path: path to image you wish to compress.
        """
        c = cls(cv2.imread(image_path))
        c._src_path = image_path
        return c

    @classmethod
    def from_image_codex(cls, image_codex: np.ndarray):
        """
        Initializes the Compressor class.

        :param image_codex: The encoded image. Intended for use when interfacing with the database.
        """
        if isinstance(image_codex, bytes):
            image_codex = np.asarray(bytearray(image_codex), dtype="uint8")
        return cls(cv2.imdecode(image_codex, cv2.IMREAD_UNCHANGED))

    def compress_to_jpg(self, jpeg_quality=95, dst_path: str = None) -> str:
        """
        Creates a jpeg version of the image using opencv's imwrite.

        :param jpeg_quality: Quality from 0 to 100, the higher the better the image but greater the size (default 95).
        :param dst_path: Must have extension .jpg. Can set to None and the function will auto generate the value.
        :return: dst_path
        """
        if dst_path is None:
            if self._src_path is None:
                dst_path = "tmp.jpg"
            else:
                dst_path = os.path.splitext(self._src_path)[0] + '_dst.jpg'

        assert os.path.splitext(dst_path)[1].lower() == '.jpg'

        # opencv will do the compression/encoding for us as long as the dst file path has the .jpg extension.
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        cv2.imwrite(dst_path, self.__image__, encode_param)

        return dst_path

    def compress_png(self, compression_level=3, dst_path: str = None) -> str:
        """
        Creates a compressed png version of the image using opencv's imwrite.
        Turns out that libpng/opencv is not very good at compression so the end file ends
        up sometimes being bigger then the original, depending on who compressed the original.

        :param src_path: File path of image.
        :param compression_level: Level from 0 to 9. The higher, the better the compression (default 3).
        :param dst_path: Must have extension .png. Can set to None and the function will auto generate the value.
        :return: dst_path
        """
        if dst_path is None:
            if self._src_path is None:
                dst_path = "tmp.png"
            else:
                dst_path = os.path.splitext(self._src_path)[0] + '_dst.png'

        assert os.path.splitext(dst_path)[1].lower() == '.png'

        # opencv will do the compression/encoding for us as long as the dst file path has the .png extension.
        encode_param = [cv2.IMWRITE_PNG_COMPRESSION, compression_level]
        cv2.imwrite(dst_path, self.__image__, encode_param)

        return dst_path

    def encode_to_jpg(self, jpeg_quality=95) -> np.ndarray:
        """
        Returns the JPG encoding on an image as a byte array.

        :param src_path: File path of image.
        :param jpeg_quality: Quality from 0 to 100, the higher the better the image but greater the size (default 95).
        :return: Encoding byte array
        """
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        _, buf = cv2.imencode('.jpg', self.__image__, encode_param)
        return buf

    def encode_to_png(self, jpeg_quality=95) -> np.ndarray:
        """
        Returns the JPG encoding on an image as a byte array.

        :param src_path: File path of image.
        :param jpeg_quality: Quality from 0 to 100, the higher the better the image but greater the size (default 95).
        :return: Encoding byte array
        """
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        _, buf = cv2.imencode('.jpg', self.__image__, encode_param)
        return buf
