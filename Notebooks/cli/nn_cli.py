from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from pprint import pprint

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# TODO - we need a common tf graph api
# TODO - refactor into nn_graph_api of core.api
# TODO - consolidate some of the stuff here with operationalization both with tf and keras models
# TODO - remove
def cv_check_network_read(network):
    try:
        net = cv2.dnn.readNetFromTensorflow(network)
        inp = np.random.standard_normal([224, 224, 3]).astype(np.float32)
        net.setInput(cv2.dnn.blobFromImage(inp))
        out = net.forward()

        if out:
            print("Successfully opened .pb file ")
            return True
        else:
            print("Unable to import_pd file\n")
            return False

    except (Exception, BaseException) as e:
        print(e)
        return False


def cv_load_effn(net, config):
    try:
        net = cv2.dnn.readNetFromTensorflow(net, config)
    except (Exception, BaseException) as e:
        print(e)
        return False


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def _load_graph(file_name):
    if not os.path.exists(file_name):
        raise ValueError("Specified graph file doesn't exist")
    with open(file_name, 'rb') as f:
        content = f.read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(content)

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    return graph


def list_to_file(l, output_file):
    with open(output_file, "w") as f:
        f.write("Number of nodes: %d" % len(l))
        f.write('\n')
        for item in l:
            f.write(item)
            f.write('\n')
    f.close()


def count_ops(graph):
    return graph.get_operations()


def get_ops(graph):
    return sum(1 for op in graph.get_operations())


def print_ops(ops, input_only=False):
    print("Number of operations: {}".format(len(ops)))
    if not input_only:
        for op in ops:
            print("{} => {}".format(op.name, op.values()))

    print()
    print('Sources (operations without inputs):')
    for op in ops:
        if len(op.inputs) > 0:
            continue
        print('- {0}'.format(op.name))

    print()
    print('Operation inputs:')
    for op in ops:
        if len(op.inputs) == 0:
            continue
        print('- {0:20}'.format(op.name))
    print('  {0}'.format(', '.join(i.name for i in op.inputs)))

    print()
    print('Tensors:')
    for op in ops:
        for out in op.outputs:
            print('- {0:20} {1:10} "{2}"'.format(str(out.shape),
                                                 out.dtype.name, out.name))


def get_nodes(graph, print=True):
    graph_def = _load_graph(graph).as_graph_def(add_shapes=True)
    nodes = [n for n in graph_def.node]
    if print:
        for n in nodes:
            pprint(n)
    return nodes


def rename_output(in_graph, output_graph, output_nodes, output_name='output'):
    if not os.path.exists(in_graph):
        raise RuntimeError("Graphdef file not found")

    graph_def = _load_graph(in_graph).as_graph_def()

    for node in graph_def.node:
        if output_nodes in node.name:
            node.name = output_name
    # Serialize and write to file
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(graph_def.SerializeToString())
    print("%d ops in the final graph." % len(graph_def.node))


def binary_to_text(in_graph, file):
    input_graph = _load_graph(in_graph)
    tf.train.write_graph(input_graph, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Graph analysis tool ')
    parser.add_argument(
        "--graph",
        default=r"E:\viNet_RnD\Deployment\Inference Models\Inception\Inception_v2_5_class.pb",
        type=str,
        required=False,
        help="Path to checkpoint files")
    parser.add_argument(
        "--output_nodes",
        default='InceptionV2/Predictions/Reshape_1',
        type=str,
        help="Names of output node, comma separated"
    )
    parser.add_argument(
        "--output_graph",
        default=r"E:\viNet_RnD\Deployment\Inference Models\Inception\inception_v2_5_class_renamed.pb",
        type=str,
        required=False,
        help="Output graph filename"
    )
    parser.add_argument(
        "--rename_outputs",
        default='output',
        type=str,
        help="Rename output nodes for better \
                        readability in production graph, to be specified in \
                        the same order as output_nodes")
    parser.add_argument(
        "--conf",
        default=r"E:\viNet_RnD\Object Detection\my_model\b0.config",
        type=str,
        help="object detection config file")

    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # cv_check_network_read(args.graph)
    rename_output(args.graph, args.output_graph, args.output_nodes)
    get_nodes(args.output_graph, True)
    # graph = _load_graph(args.graph)
    # print_ops(count_ops(graph), False)
