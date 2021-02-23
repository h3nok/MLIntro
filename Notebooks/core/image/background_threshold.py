import cv2
import numpy as np
import random


class BackgroundThresholder:
    """
    A wrapper class around the opencv grabCut method.

    `__variable__` - enforced
    `__variable` - required
    `_variable` - optional
    """

    __image__ = None
    __bounding_box = None
    _rng_seed = None
    # Any value from 2-10 produces similar looking images with higher numbers giving a tighter fit.
    # Note that for `__iterations = 2` the images will not have 'spots' and so `_remove_spots` can be false.
    # Also note, I want this to be hardcoded as it make using this class a lot easier.
    __iterations = 10
    # Calling remove_mask_spots is very slow so having a way to turn it off might be helpful.
    _remove_spots = True
    _outline_mode = False

    def __init__(self, image: np.ndarray):
        """
        Initializes the BackgroundThresholder class.

        :param image: The decoded image.
        """
        self.__image__ = image
        # TODO: should we shave off the alpha channel if it is there?
        # I'm going to say no for now.

    @classmethod
    def from_image_path(cls, image_path: str):
        """
        Initializes the BackgroundThresholder class.

        :param image_path: path to image you wish to threshold.
        """
        return cls(cv2.imread(image_path))

    @classmethod
    def from_image_codex(cls, image_codex: np.ndarray):
        """
        Initializes the BackgroundThresholder class.

        :param image_codex: The encoded image. Intended for use when interfacing with the database.
        """
        if isinstance(image_codex, bytes):
            image_codex = np.asarray(bytearray(image_codex), dtype="uint8")
        return cls(cv2.imdecode(image_codex))

    def set_bounding_box(self, x, y, w, h):
        """
        Sets the bounding box that will be used by grab cut.

        Anything outside of the box will be marked as a guaranteed background.

        :param x: Top left corner x-pos
        :param y: Top left corner y-pos
        :param w: Width
        :param h: Height
        """
        self.__bounding_box = (x, y, w, h)

    def set_rng_seed(self, seed: int):
        """
        This is intended for use by test functions.
        It turns out that grabCut uses some amount of rng and in order to insure test consistency we need to tell
        openCV that we want to use a constant seed.

        :param seed: the seed
        """
        self._rng_seed = seed

    def set_outline_mode(self):
        """
        This is intended to be use for visual tests.

        Once set, when you call `grab_foreground` it will create an image with the foreground outlined.
        """
        self._outline_mode = True

    def __set_background(self, mask, background_color):
        """
        INTERNAL FUNCTION: Not intended for external use.

        Takes in a 0,1 mask where 0 is background and 1 foreground. Will set background to be `background_color`

        :param mask: 0,1 mask
        :param background_color: The intended background color. If (-1, -1, -1) then will take average existing color.
        Add a 4th value for alpha (0 - full transparency, 255 - no transparency).
        :return: new masked_img
        """
        inv_mask = (cv2.bitwise_not(mask) / 255).astype('uint8')

        chans = len(background_color)

        assert chans == 3 or chans == 4

        if background_color[0:3] == (-1, -1, -1):
            background_color = cv2.mean(self.__image__, inv_mask)[0:chans]

        background = np.zeros((self.__image__.shape[0], self.__image__.shape[1], chans), np.uint8)
        background[:, :] = background_color
        background = background * inv_mask[:, :, np.newaxis]

        img = self.__image__
        if chans == 4:
            # We want to add a 4th channel to the original image.
            fg_b_channel, fg_g_channel, fg_r_channel = cv2.split(img)
            fg_a_channel = np.ones(fg_b_channel.shape, dtype=fg_b_channel.dtype) * 255
            img = cv2.merge((fg_b_channel, fg_g_channel, fg_r_channel, fg_a_channel))

        masked_img = img * mask[:, :, np.newaxis]
        masked_img = masked_img + background

        return masked_img

    def __outline_foreground(self, mask, background_color):
        """
        INTERNAL FUNCTION: Not intended for external use.
        This is so we can better see what it grabbed and what is not from the original image.

        Takes in a 0,1 mask where 0 is background and 1 foreground. Will set the outline to be `background_color`

        :param mask: 0,1 mask
        :param background_color: The intended background color. If (-1, -1, -1) then will take average existing color.
        Add a 4th value for alpha (0 - full transparency, 255 - no transparency).
        :return: new masked_img
        """
        # we will find the group of pixels that is the foreground object and then create an edge around that.
        outline_mask = np.ones(mask.shape, np.uint8)

        # find starting spot
        if self._rng_seed is not None:
            random.seed(self._rng_seed)
        start_row, start_col = (mask.shape[0]//2, mask.shape[1]//2)
        loops = 0
        while mask[start_row, start_col] != cv2.GC_FGD:
            # This is such a dumb way to do this
            if loops > mask.shape[0] * mask.shape[1]:
                raise BaseException("Failed to find and outline the foreground object.")
            start_row, start_col = (random.randrange(mask.shape[0]), random.randrange(mask.shape[1]))

        # do ms paint bucket thing
        # We start at a foreground pixel and continually expand the foreground zone until we reach a background pixel.
        # From there we set that pixel in the `outline_mask` to be an edge. This will create a 1 pixel thick bound of
        # the foreground object.
        q = []
        hmask = np.zeros(mask.shape, np.uint8)
        hmask[start_row, start_col] = 1
        q.append((start_row, start_col))

        while len(q) > 0:
            r, c = q.pop(0)
            for i in range(-1, 2):  # -1, 0, 1
                for j in range(-1, 2):  # -1, 0, 1
                    if r + i < 0 or c + j < 0 or r + i >= mask.shape[0] or c + j >= mask.shape[1]:
                        continue
                    elif hmask[r + i, c + j] == 0 and mask[r + i, c + j] == cv2.GC_FGD:
                        hmask[r + i, c + j] = 1
                        q.append((r + i, c + j))
                    elif mask[r + i, c + j] == cv2.GC_BGD:
                        # any pixel next to the foreground will be the edge
                        outline_mask[r + i, c + j] = cv2.GC_BGD

        return self.__set_background(outline_mask, background_color)

    # It might be worth writing this in c++ for better performance
    @staticmethod
    def __smooth_mask(mask: np.ndarray) -> np.ndarray:
        """
        NOT USED.
        INTERNAL FUNCTION: Not intended for external use.

        Take in a 0,1,2,3 mask and "smooth" it
        we just want to get rid of stray possible backgrounds in the foreground object
        Note: cv2.GC_BGD = 0, cv2.GC_FGD = 1, cv2.GC_PR_BGD = 2, cv2.GC_PR_FGD = 3

        The point of this function is to fix the case where parts of the bird are marked as 2 (probable background).
        An example of this can be seen with `test_grab_foreground3_remove_spots`.

        Here I am trying to solve this problem by using cellular automata. But, it doesn't seem like this is working.
        well I think we can make it work if we add the paint brush method of using grabcut.

        :param mask: The mask returned by grab cut.
        :return: new_mask
        """

        def reeval_point(r, c) -> int:
            bg_neighbors = 0
            for i in range(-1, 2):  # -1, 0, 1
                for j in range(-1, 2):
                    if i == j == 0 or c + i < 0 or c + i >= mask.shape[1] or r + j < 0 or r + i >= mask.shape[0]:
                        pass
                    elif mask[r + j, c + i] == cv2.GC_BGD:  # 0: background
                        return cv2.GC_PR_BGD  # Is this what I want
                    elif mask[r + j, c + i] == cv2.GC_FGD:  # 1: foreground
                        return cv2.GC_PR_FGD
                    elif mask[r + j, c + i] == cv2.GC_PR_BGD:  # 2: probable background
                        bg_neighbors += 1
            if bg_neighbors <= 2:
                return cv2.GC_PR_FGD
            elif bg_neighbors >= 6:
                return cv2.GC_PR_BGD
            else:
                return mask[r, c]

        new_mask = np.copy(mask)

        for row in range(0, mask.shape[0]):
            for col in range(0, mask.shape[1]):
                if mask[row, col] == cv2.GC_PR_BGD or mask[row, col] == cv2.GC_PR_FGD:
                    new_mask[row, col] = reeval_point(row, col)

        # print(np.all(mask == new_mask))
        return new_mask

    # It might be worth writing this in c++ for better performance
    @staticmethod
    def __remove_mask_spots(mask: np.ndarray) -> np.ndarray:
        """
        INTERNAL FUNCTION: Not intended for external use.

        Takes in a 0,1,2,3 mask and "smooth" it
        we just want to get rid of stray 'possible backgrounds' (mask value 2) inside the foreground object.
        Note: cv2.GC_BGD = 0, cv2.GC_FGD = 1, cv2.GC_PR_BGD = 2, cv2.GC_PR_FGD = 3

        The point of this function is to fix the case where parts of the bird are marked as 2 (probable background).
        An example of this can be seen with `test_grab_foreground3_remove_spots`.

        I don't know what the algorithm is called but it functions the same way that the bucket does in ms paint.
        I guess breath first search.

        :param mask: The mask returned by grab cut.
        :return: mask after changes
        """

        min_allowed_group_size = (mask.shape[0] * mask.shape[1]) / 100  # TODO: does 100 work well?

        hmask = np.zeros(mask.shape, np.uint8)
        group_id = 1

        def create_group(row, col, gid):
            """
            I don't know what the algorithm is called but it functions the same way that the bucket does in ms paint.
            I guess breath first search.

            :param row: initial starting point
            :param col: initial starting point
            :param gid: must not be 0 (other then that, it is not used)
            :return:
            """

            group = []
            q = []
            hmask[row, col] = gid
            group.append((row, col))
            q.append((row, col))
            is_def_bg = False
            # is_def_fg = False

            while len(q) > 0:
                r, c = q.pop(0)
                for i in range(-1, 2):  # -1, 0, 1
                    for j in range(-1, 2):  # -1, 0, 1
                        if r + i < 0 or c + j < 0 or r + i >= mask.shape[0] or c + j >= mask.shape[1]:
                            continue
                        elif hmask[r + i, c + j] == 0 and mask[r + i, c + j] == cv2.GC_PR_BGD:
                            hmask[r + i, c + j] = gid
                            group.append((r + i, c + j))
                            q.append((r + i, c + j))
                        elif mask[r + i, c + j] == cv2.GC_BGD:
                            is_def_bg = True

            return group, is_def_bg

        for row in range(0, mask.shape[0]):
            for col in range(0, mask.shape[1]):
                if mask[row, col] == cv2.GC_PR_BGD and hmask[row, col] == 0:  # we are probable bg and not in group yet
                    # we want to create this group
                    group, is_def_bg = create_group(row, col, group_id)
                    group_id += 1

                    if len(group) > min_allowed_group_size or is_def_bg:
                        pass  # stay background
                    else:
                        # print("mask edited")
                        for (r, c) in group:
                            mask[r, c] = cv2.GC_PR_FGD  # 3: probable fg

        return mask

    def grab_foreground(self, dst_path: str, background_color=(0, 0, 0)) -> any:
        """
        Will create an image with the foreground extracted out.
        `set_bounding_box` must be called before calling this function.

        :param dst_path: File to save the resulting image to. Can be None.
        :param background_color: The intended background color. If (-1, -1, -1) then will take average existing color.
        Add a 4th value for alpha (0: full transparency, 255: no transparency).
        :return: if `dst_path is None` then will return `np.ndarray` of image `else` will return `dst_path`
        """

        assert self.__bounding_box is not None
        assert self.__image__ is not None
        assert self.__image__.shape[2] == 3  # no alpha channel allowed

        mask = np.zeros(self.__image__.shape[:2], np.uint8)

        # These are arrays used by the algorithm internally.
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        if self._rng_seed is not None:
            cv2.setRNGSeed(self._rng_seed)

        # I don't think that doing it this way i.e. with two calls is any different from the original single call,
        # but when I looked op opencv's test code they did it this way.
        cv2.grabCut(self.__image__, mask, self.__bounding_box, bgd_model, fgd_model, 0,
                    mode=cv2.GC_INIT_WITH_RECT)
        cv2.grabCut(self.__image__, mask, self.__bounding_box, bgd_model, fgd_model, self.__iterations,
                    mode=cv2.GC_EVAL)

        if self._remove_spots is not None and self._remove_spots:
            # mask = self.__smooth_mask(mask)
            mask = self.__remove_mask_spots(mask)

        # The mask is a matrix where each entry represents what it thinks that pixel is on the original image.
        # the possible mask values are 0: background, 1: foreground, 2: probable background, 3: probable foreground.
        # pixels will only be marked 0 or 1 is we tell the program that something if foreground or background e.g.
        # we told it that everything outside of the background is background.
        # Therefor we want to make a new mask where both types of background are masked out.
        # Note: cv2.GC_BGD = 0, cv2.GC_FGD = 1, cv2.GC_PR_BGD = 2, cv2.GC_PR_FGD = 3
        mask = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype('uint8')

        if self._outline_mode:
            masked_img = self.__outline_foreground(mask, background_color)
        else:
            masked_img = self.__set_background(mask, background_color)

        # cv2.imshow("win", masked_img)
        # cv2.waitKey(0)

        if dst_path is None:
            return masked_img
        else:
            cv2.imwrite(dst_path, masked_img)
            return dst_path
