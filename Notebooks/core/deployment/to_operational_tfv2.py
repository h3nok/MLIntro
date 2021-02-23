import os

import tensorflow as tf

# This is needed, don't remove

from transform.graph_transforms import transform_graph
from core.api.nn_graph_api import KerasModelWrapper


def h5_to_save_model(model_file, model_out_dir):
    """
    `to_graph_def_protobuf_file` has worked better when the input is a .h5 file so this is not needed.

    :param model_file:
    :param model_out_dir:
    :return:
    """
    model = tf.keras.models.load_model(model_file)
    model.save(model_out_dir, save_format='tf')


def operationalize(model_file, out_graph_file, input_node_names='input',
                   output_node_names='Identity', clean=True) -> None:
    """
    Takes a keras model file (.h5) and turns it into an operation ready graph that OpenCV can read.
    Calls to_graph_def_protobuf_file, optimize, and transform_graph.
    Note, transform_graph requires WSL.
    Here is a setup guide: `\\qa\tmp\1zjorquera\WSL_tf\set_up.md`.

    @param model_file: in the format .h5
    @param input_node_names: Most likely should be 'input_1'
    @param output_node_names: Most likely should be 'Identity'
    @param out_graph_file: format .pb
    @return:
    @param clean: remove artifacts after conversion
    """
    freeze_output = os.path.splitext(out_graph_file)[0] + '_after_freeze' + os.path.splitext(out_graph_file)[1]
    optimize_output = os.path.splitext(out_graph_file)[0] + '_after_optimize' + os.path.splitext(out_graph_file)[1]

    # assert not os.path.exists(out_graph_file), out_graph_file
    # assert not os.path.exists(freeze_output)
    # assert not os.path.exists(optimize_output)

    if not os.path.exists(os.path.dirname(out_graph_file)):
        os.makedirs(os.path.dirname(out_graph_file))
    keras_model = KerasModelWrapper(model_file)
    img_shape, input_node_str = keras_model.freeze(freeze_output)
    # to_graph_def_protobuf_file(model_file, freeze_output)
    assert output_node_names in keras_model.output_layers

    # optimize(freeze_output, input_node_names, output_node_names, optimize_output)
    keras_model.optimize(input_node_names, output_node_names, optimize_output, False)

    transforms = ['remove_nodes(op=PlaceholderWithDefault)',
                  'strip_unused_nodes(type=float, '
                  'shape="1,{},{},3")'.format(img_shape[0], img_shape[1]),
                  'sort_by_execution_order']
    if transform_graph(optimize_output, input_node_names, output_node_names, transforms, out_graph_file) is None:
        print('Failed to call transform_graph through wsl: skipping step.')

    if clean:
        os.remove(freeze_output)
        os.remove(optimize_output)
        os.remove(model_file)


def print_layers(model_path):
    assert os.path.exists(model_path)
    # assert os.path.splitext(model_path)[1] == '.h5'

    model = tf.keras.models.load_model(model_path)

    model.summary()
