from unittest import TestCase
from deployment.to_operational_tfv2 import *
from transform.test_graph_transforms import cv_check_network_read, get_all_like_files

import cv2

MIN_CV_VER = [4, 3, 0]


class Test(TestCase):
    def test_to_operational(self):
        input_model_path = r'\\qa\tmp\1zjorquera\efficientnet\saves\v2\take2\done.h5'
        output_path = r'E:\viNet_RnD\Deployment\Inference Models\Efn\efficientnet_b0.pb'

        input_node_names = 'input'  # By construction, input is always the default input.
        # look at `to_graph_def_protobuf_file`
        output_node_names = 'efficientnet-b0/probs/Softmax'  # or could be 'efficientnet-b0/probs/Softmax' maybe

        # Note, to_operational requires WSL.
        # Here is a setup guide: `\\qa\tmp\1zjorquera\WSL_tf\set_up.md`.
        operationalize(input_model_path, output_path, input_node_names, output_node_names)

        # We cant test opencv read with an openCV version above or equal to 4.3.0 (MIN_CV_VER)
        if [int(v) >= MIN_CV_VER[i] for i, v in enumerate(cv2.__version__.split('.')[:3])].count(True) == 3:
            assert cv_check_network_read(output_path)

        # # clean up
        for file in get_all_like_files(os.path.basename(output_path), os.path.dirname(output_path)):
            os.remove(file)

    def test(self):
        input_model_path = r"E:\viNet_RnD\Deployment\Inference Models\Efn\done.h5"
        print_layers(input_model_path)
