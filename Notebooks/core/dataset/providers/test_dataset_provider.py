from unittest import TestCase
from providers.reportdata_provider import viNetResultDataProvider
from providers.framedata_provider import PixelFrameProvider
from frame_datum import FrameDatum
import random


class TestviNetResultDataProvider(TestCase):
    def test(self):
        tag = 'viNet_2.7_Vattenfall_4_class_6.3m'
        config = 'Vattenfall-Gotland Training V2 (4 class) - Validation'
        classification_group = 'Vattenfall Proper 3 Class'
        procedure = 'get_network_classification_results_on_grouped_frames'
        dp = viNetResultDataProvider(tag, config,
                                     classification_group=classification_group,
                                     procedure=procedure)
        dp.set_limit(1)
        dp.fetch_results()
        print(dp.dataframe.vi.columns)


class TestPixelFrameProvider(TestCase):
    def test(self):
        with PixelFrameProvider() as pfp:
            frame = pfp.fetch_one('Golden-Eagle')
            frame.show()
            assert frame
            assert isinstance(frame, FrameDatum)

            categories = ['Golden-Eagle', 'Raven']

            frames = pfp.fetch_many(categories, limit=20)
            assert isinstance(frames, list)
            random.shuffle(frames)
            for frame in frames:
                print(frame.groundtruth, frame.shape)

