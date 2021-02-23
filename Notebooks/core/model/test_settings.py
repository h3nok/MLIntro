from unittest import TestCase
from training_config import Config, NeuralNetConfig


class TestviNetConfigs(TestCase):
    def test_viNetConfig(self):
        test_file = 'test.ini'

        test_config = Config(test_file)
        test_config.dump_data()

    def test_EfficientNetSettings(self):
        test_file = 'test.ini'

        test_config = NeuralNetConfig(test_file)
        test_config.dump_data()

