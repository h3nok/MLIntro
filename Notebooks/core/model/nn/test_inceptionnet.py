from unittest import TestCase
# from inception import InceptionNet
from training_config import NeuralNetConfig
from customer.customers import *
from neuralnet import NeuralNet


class TestInceptionNet(TestCase):
    config = NeuralNetConfig('vinet.settings.test.ini')

    def test_inception_v3(self):
        self.config.set_net('InceptionV3')
        self.config.set_input_shape((75, 75, 3))
        net = NeuralNet(self.config, CustomerObjectMap['GWA'])
        print(net.summary)

        net.train()

    def test_xception(self):
        self.config.set_net('Xception')
        self.config.set_input_shape((71, 71, 3))  # the smallest size accepted by Xception
        net = NeuralNet(self.config, GWA)
        print(net.summary)
        net.train()

    def test_inception_resnet(self):
        self.config.set_net('InceptionResNet')
        self.config.set_input_shape((75, 75, 3))
        net = NeuralNet(self.config, GWA)
        print(net.summary)
        # net.train()
        net.save()
