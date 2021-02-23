from unittest import TestCase
from database.database_interface import PgsqlInterface
import sys, os


class TestPgsqlInterface(TestCase):
    def test_read_sproc(self):
        server = PgsqlInterface()
        limit  = 10
        server.connect('../../.config/database_h.ini')
        df = server.read_sproc(sp='get_network_classification_results_with_images',
                               tag='E3_v2.2 Verification',
                               config='viNet_2.2_E3_10M', limit=limit)
        assert not df.empty
        assert len(df.index) == limit
        assert df.shape[1] > 0

        df = server.read_sproc(sp='get_network_classification_results_on_grouped_frames',
                               config='viNet_2.7_Vattenfall_4_class_6.3m',
                               tag='Vattenfall-Gotland Training V2 (4 class) - Validation',
                               group='Vattenfall Proper 4 Class', limit=limit)

        assert not df.empty
        assert df.shape[1] > 0
        assert len(df.index) == limit
