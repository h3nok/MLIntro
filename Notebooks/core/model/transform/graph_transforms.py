import os

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

from wsl_executable import to_wsl_path, WslExecutable

USE_OPTIMIZE_FOR_INFERENCE_LIB = False
FLAGS = None

TRANSFORM_GRAPH_DIR = r'/home/tensorflow/bazel-bin/tensorflow/tools/graph_transforms'


def freeze(input_graph_path, input_checkpoint_dir, output_node_names, output_path, binary_input=True,
           checkpoint_file=None):
    """
    Calls the tensorflow freeze_graph python tool

    :param input_graph_path:
    :param input_checkpoint_dir:
    :param output_node_names: of the graph
    :param output_path:
    :param binary_input:
    :param checkpoint_file: to choose a specific checkpoint not just the latest.
    :return: edited output_path
    """
    assert os.path.exists(input_graph_path)
    assert os.path.exists(input_checkpoint_dir)
    if checkpoint_file is None:
        checkpoint_path = tf.train.latest_checkpoint(input_checkpoint_dir)
    else:
        checkpoint_path = os.path.join(input_checkpoint_dir, checkpoint_file)

    # default value
    input_saver = ""
    # restore_op_name = "save/restore_all"  # Deprecat
    restore_op_name = None  # Deprecat
    # filename_tensor_name = "save/Const:0"  # Deprecated
    filename_tensor_name = None  # Deprecated
    clear_devices = True
    initializer_nodes = ""

    freeze_graph.freeze_graph(input_graph_path, input_saver, binary_input, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name, output_path, clear_devices, initializer_nodes)

    return output_path


def optimize(input_graph_path, input_node_names, output_node_names, output_path):
    """
    calls the tensorflow optimize_for_inference python tool through optimize_for_inference_lib

    :param input_graph_path:
    :param input_node_names:
    :param output_node_names:
    :param output_path:
    :return: output_path
    """
    input_graph_def = graph_pb2.GraphDef()
    with tf.io.gfile.GFile(input_graph_path, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    if USE_OPTIMIZE_FOR_INFERENCE_LIB:
        output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names.split(","), output_node_names.split(","),
            dtypes.float32.as_datatype_enum)
    else:
        output_graph_def = tf.compat.v1.graph_util.remove_training_nodes(input_graph_def)

    f = tf.io.gfile.GFile(output_path, "w")
    f.write(output_graph_def.SerializeToString())

    return output_path


def transform_graph(input_graph_path, input_node_names, output_node_names, transforms, output_path):
    """
    calls the tensorflow transform_graph c++ tool though wsl.
    requires WSL on the current computer and for transform_graph to be built in WSL.
    Here is a setup guide: \\qa\tmp\1zjorquera\WSL_tf\set_up.md

    :param input_graph_path:
    :param input_node_names:
    :param output_node_names:
    :param transforms:
    :param output_path: Must by on the current system. Can't be at \\qa or something.
    :return: output_path
    """
    # calls the WSL command `transform_graph`
    try:
        wsl_exe = WslExecutable('{}/transform_graph'.format(TRANSFORM_GRAPH_DIR))
    except AssertionError:
        return None

    output_path_win = output_path
    input_graph_path = to_wsl_path(input_graph_path)
    output_path = to_wsl_path(output_path)
    # assert os.path.exists(input_graph_path), f"{input_graph_path}"
    if not isinstance(input_node_names, str):
        inn_str = ''
        for inn in input_node_names:
            inn_str += inn + ' '
    else:
        inn_str = input_node_names

    if not isinstance(output_node_names, str):
        onn_str = ''
        for onn in output_node_names:
            onn_str += onn + ' '
    else:
        onn_str = output_node_names

    if not isinstance(transforms, str):
        transforms_str = ''
        for transform in transforms:
            transforms_str += transform + ' '
    else:
        transforms_str = transforms

    transform_graph_args = [r"--in_graph='{}'".format(input_graph_path),
                            r"--out_graph='{}'".format(output_path.strip()),
                            r"--inputs='{}'".format(inn_str),
                            r"--outputs='{}'".format(onn_str),
                            r"--transforms='{}'".format(transforms_str)]

    print(f"Executing transform_graph, args: {transform_graph_args}")
    wsl_exe.run(transform_graph_args)
    _, out = wsl_exe.pipe_output(block=True)

    # assert os.path.exists(output_path_win), out
    print(out)

    return output_path


def operationalize(input_graph_path, input_checkpoint_dir, input_node_names, output_node_names, output_path,
                   image_shape, is_binary=True, checkpoint_file=None, cleanup=True):
    """
    Calls freeze_graph, optimize_for_inference, and transform_graph.
    Note, transform_graph requires WSL.
    Here is a setup guide: `\\qa\tmp\1zjorquera\WSL_tf\set_up.md`.

    :param input_graph_path:
    :param input_checkpoint_dir:
    :param input_node_names:
    :param output_node_names:
    :param output_path:
    :param image_shape:
    :param is_binary:
    :param checkpoint_file:
    :return: output_path
    """
    freeze_output = os.path.splitext(output_path)[0] + '_after_freeze' + os.path.splitext(output_path)[1]
    optimize_output = os.path.splitext(output_path)[0] + '_after_optimize' + os.path.splitext(output_path)[1]

    assert not os.path.exists(output_path)
    assert not os.path.exists(freeze_output)
    assert not os.path.exists(optimize_output)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    freeze_output = freeze(input_graph_path, input_checkpoint_dir,
                           output_node_names, freeze_output, is_binary,
                           checkpoint_file)

    optimize_output = optimize(freeze_output, input_node_names,
                               output_node_names, optimize_output)
    # optimize_output = _optimize(freeze_output, input_names, output_names, output_path)

    # c++ thing
    transforms = ['remove_nodes(op=PlaceholderWithDefault)',
                  'strip_unused_nodes(type=float, shape="1,{},{},3")'.format(image_shape[0], image_shape[1]),
                  'sort_by_execution_order']
    assert os.path.exists(optimize_output)
    if transform_graph(optimize_output.strip(), input_node_names, output_node_names,
                       transforms, output_path) is None:
        print('Failed to call transform_graph through wsl: skipping step.')
        # os.rename(optimize_output, output_path)

    if cleanup:
        os.remove(freeze_output)
        os.remove(optimize_output)

    return output_path
