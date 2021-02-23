import unittest
# from inception import InceptionNet
from training_config import NeuralNetConfig
from customer.customers import *
from neuralnet import NeuralNet


class TestMobileNet(unittest.TestCase):
    config = NeuralNetConfig('vinet.settings.test.ini')

    def test_MobileNet(self):
        self.config.set_net('MobileNet')
        net = NeuralNet(self.config, GWA)
        print(net.summary)
        net.train()

    def testMobileNetV2(self):
        self.config.set_net('MobileNetV2')
        net = NeuralNet(self.config, GWA)
        print(net.summary)
        net.train()


