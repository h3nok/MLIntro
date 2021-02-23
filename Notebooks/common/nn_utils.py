import netron
import os
import tensorflow as tf


def visualize_graph(model_path):
    """

    @param model_path:
    @return:
    """
    assert model_path
    return netron.start(model_path)


def load_pb_graph(file_name):
    """

    @param file_name:
    @return:
    """
    if not os.path.exists(file_name):
        raise ValueError("Specified graph file doesn't exist")
    with open(file_name, 'rb') as f:
        content = f.read()
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(content)

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    return graph


def print_layers(model_file_path, more=True):
    """
    Prints the nodes of a .pb graph
    @param model_file_path:
    @param more:
    @return:
    """
    tf.compat.v1.disable_v2_behavior()

    with tf.io.gfile.GFile(model_file_path, 'rb') as f:
        global_graph_def = tf.compat.v1.GraphDef()
        global_graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    sess = tf.compat.v1.InteractiveSession(graph=graph)
    tf.import_graph_def(global_graph_def)
    ops = graph.get_operations()

    if more:
        layers = ['{0:20} {1:10} "{2}"'.format(str(out.shape), out.dtype.name, out.name)
                  for op in ops for out in op.outputs]
    else:
        layers = [op.name for op in ops]

    for l in layers:
        print(l)


