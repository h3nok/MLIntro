import tensorflow as tf


def wrap_frozen_graph(graph_def, inputs, outputs):
    """
    Takes a frozen graph def and turns is into something tfv2 can work with (not a graph).

    :param graph_def:
    :param inputs: input layers, can be a list
    :param outputs: output layers, can be a list
    :return: a wrapped function
    """
    assert outputs is not None

    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


