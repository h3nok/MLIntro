import unittest
# from inception import InceptionNet
from training_config import NeuralNetConfig
from customers import *
from neuralnet import NeuralNet


class TestMobileNet(unittest.TestCase):
    config = NeuralNetConfig('model/nn/vinet.settings.test.ini')

    def testNASNetMobile(self):
        self.config.set_net('NASNetMobile')
        net = NeuralNet(self.config, GWA)
        print(net.summary)
        net.train()

    def testDenseNet121(self):
        self.config.set_net('DenseNet121')
        net = NeuralNet(self.config, GWA)
        print(net.summary)
        net.train()

    def testDenseNet169(self):
        self.config.set_net('DenseNet169')
        net = NeuralNet(self.config, GWA)
        print(net.summary)
        net.train()

