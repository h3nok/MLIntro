from unittest import TestCase
from training_config import NeuralNetConfig
from neuralnet import NeuralNet
import tensorflow as tf
import numpy as np
import cv2
from customer.customers import *

from sklearn.metrics import confusion_matrix, accuracy_score

TEST_IMAGE_PATH = r'../../image/test_images/test_image2.png'


class TestEfficientNet(TestCase):
    def test_train(self):
        config = NeuralNetConfig('vinet.settings.test.ini')
        version = ['B0', 'B1', 'B2', 'B3']
        for v in version:
            config.set_net(v)
            # config.dump_data() # call this function to update the ini file anytime you add a new param to settings
            net = NeuralNet(config, GWA)
            # print(net.summary)
        # net.train()
        # net.plot()
        # net.save_model()
        # print(net.evaluate())

    def test_load_weights(self):
        efn_settings = NeuralNetConfig('model/nn/vinet.settings.test.ini')

        net = NeuralNet(efn_settings)
        net.load_weights(r'\\qa\tmp\1zjorquera\efficientnet\saves\v2\take2\cp.ckpt-e05')
        net.save()

        print(net.evaluate())

    def test_resize_input_image(self):
        image = cv2.imread(TEST_IMAGE_PATH)
        tfimage = tf.constant(image)

        efn_settings = NeuralNetConfig('model/nn/vinet.settings.test.ini')

        net = NeuralNet(efn_settings)
        rs_image = net.resize_input_image(image)
        rs_tfimage = net.resize_input_image(tfimage)
        assert isinstance(rs_tfimage, tf.Tensor)
        assert isinstance(rs_image, np.ndarray)

        min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(np.reshape(rs_image, -1))
        assert min_val >= -1
        assert max_val <= 1
        assert rs_image.shape == net.input_shape

    def test_predict_validation(self):
        efn_settings = NeuralNetConfig('model/nn/vinet.settings.test.ini')

        net = NeuralNet(efn_settings)
        net.load_weights(r'\\qa\tmp\1zjorquera\efficientnet\saves\v2\take2\done.ckpt')

        ids, truth, pred = net.predict_validation(hard_labels=True)

        acc = accuracy_score(truth, pred)

        bgt = [v == 1 for v in truth]  # note 1 is eagle
        bpred = [v == 1 for v in pred]

        tn, fp, fn, tp = confusion_matrix(bgt, bpred).ravel()
        tpr = (tp / (tp + fn))  # tpr
        tnr = (tn / (tn + fp))  # tnr
        fpr = (fp / (fp + tn))  # fpr
        fnr = (fn / (fn + tp))  # fnr

        bacc = accuracy_score(bgt, bpred)

        print("acc:", acc, "bacc:", bacc, "tpr:", tpr, "tnr:", tnr, "fpr:", fpr, "fnr:", fnr)
