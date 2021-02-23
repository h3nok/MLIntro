from unittest import TestCase
from transform.graph_transforms import operationalize

import cv2
import numpy as np
import regex as re
import os


def cv_check_network_read(network):
    try:
        net = cv2.dnn.readNetFromTensorflow(network)
        inp = np.random.standard_normal([224, 224, 3]).astype(np.float32)
        net.setInput(cv2.dnn.blobFromImage(inp))
        out = net.forward()
        if out is not None:
            return True
        else:
            print("Unable to import_pd file\n")
            return False
    except (Exception, BaseException) as e:
        print(e)
        return False


def get_all_like_files(file_name, dir_to_clean):
    like_files = [
        os.path.join(dir_to_clean, fn)
        for fn in next(os.walk(dir_to_clean))[2]
        if re.match(os.path.splitext(file_name)[0] + '_.*' + os.path.splitext(file_name)[1], fn)
    ]
    like_files.append(os.path.join(dir_to_clean, file_name))
    return like_files


class Test(TestCase):
    def test_to_operational(self):
        # input_graph_path = r'C:\viNet_RnD\Training\jpg\Output\Vattenfall-Gotland\inception_v2_2020_06_10_07_5028
        # \graph.pbtxt' input_checkpoint_dir =
        # r'C:\viNet_RnD\Training\jpg\Output\Vattenfall-Gotland\inception_v2_2020_06_10_07_5028'
        # input_checkpoint_file = r'model.ckpt-2911210' output_path = r'C:\viNet_RnD\Deployment\Frozen
        # Models\Inception_v2_frozen_test.pb' is_binary = False
        input_graph_path = r"E:\viNet_RnD\Deployment\Inference Models\Inception\inception_v2_6_class_renamed.pb"
        input_checkpoint_dir = r'E:\viNet_RnD\Deployment\Vattenfall\v4'
        input_checkpoint_file = r'model.ckpt-844723'
        output_path = r'E:\viNet_RnD\Deployment\Vattenfall\v4\viNet_2.9_3M.pb'
        is_binary = True

        input_node_names = 'input'
        output_node_names = 'output'

        image_width_height = 224

        # Note, to_operational requires WSL.
        # Here is a setup guide: `\\qa\tmp\1zjorquera\WSL_tf\set_up.md`.
        operationalize(input_graph_path, input_checkpoint_dir,
                       input_node_names,
                       output_node_names, output_path,
                       (image_width_height, image_width_height),
                       checkpoint_file=input_checkpoint_file,
                       is_binary=is_binary)

        assert cv_check_network_read(output_path)

        # # clean up
        # for file in get_all_like_files(os.path.basename(output_path), os.path.dirname(output_path)):
        #     os.remove(file)

    def test(self):
        output_path = r'C:\viNet_RnD\Deployment\Frozen Models\Inception_v2_frozen_test1.pb'
        output_path2 = r'C:\viNet_RnD\Deployment\Frozen Models\Efficientnet_test.pb'
        # print_layers(output_path2)
        pass
