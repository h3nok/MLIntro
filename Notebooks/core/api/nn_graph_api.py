import os

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.keras.models import load_model
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.tools import optimize_for_inference_lib

from transform.graph_transforms import transform_graph


class ProtobufWrapper:
    def __init__(self, pb_file_path, load=False):
        """
        TODO - context manager
        @param pb_file_path:
        @param load:
        """
        assert pb_file_path
        self.__graph_path = pb_file_path
        assert os.path.exists(self.__graph_path)
        self.__graph_def = None
        self.__graph = None  # in memory graph, can manipulate layers etc
        self.__ops = None
        self._load()

        print(f"nodes: {len(self.nodes)}, ops: {len(self.ops)}")

    @property
    def nodes(self):
        return self.__graph_def.node

    @property
    def ops(self):
        if not self.__ops or len(self.__ops) == 0:
            self.__ops = self.__graph.get_operations()
        return self.__ops

    def _load(self):
        """
        Loads the protobuf file to memory
        """
        self.__graph_def = None
        self.__graph = None
        self.__ops = None

        if not os.path.exists(self.__graph_path):
            raise ValueError("Specified graph file doesn't exist")

        with open(self.__graph_path, 'rb') as f:
            content = f.read()
            self.__graph_def = tf.compat.v1.GraphDef()
            self.__graph_def.ParseFromString(content)

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(self.__graph_def, name='')

        print(f"Successfully loaded \'{self.__graph_path}\'")
        self.__graph = graph

    def layers(self, ops=None, num_classes=3):
        if not ops:
            ops = self.ops

        output_layers = []

        for op in ops:
            for out in op.outputs:
                if str(num_classes) in str(out.shape):
                    output_layers.append(op.name)

        return output_layers

    def rename_output_layer(self, output_graph, output_node, new_name='output'):
        for node in self.nodes:
            if output_node in node.name:
                node.name = new_name
                print(f"Successfully renamed \'{output_node}\' to \'{new_name}\'")

        # Serialize and write to file
        with tf.io.gfile.GFile(output_graph, "wb") as f:
            f.write(self.__graph_def.SerializeToString())

        self.__graph_path = output_graph
        self._load()

        # print("%d ops in the final graph." % len(self.nodes))


class KerasModelWrapper:

    def __init__(self, graph_path):
        """

        @param graph_path:
        """
        assert graph_path
        assert 'h5' in graph_path
        assert os.path.exists(graph_path)

        self._graph_path = graph_path

        self.__model = load_model(self._graph_path)
        self.__layers = self.__model.layers
        self.__frozen_model = None
        self.__output_layers = None
        self.__optimized_model = None

    @property
    def summary(self):
        return print(self.__model.summary())

    @property
    def config(self):
        return self.__model.get_config()

    @property
    def frozen(self):
        return self.__frozen_model

    @property
    def optimized(self):
        return self.__optimized_model

    @property
    def output_layers(self):
        if not self.__output_layers:
            g = ProtobufWrapper(self.__frozen_model)
            # g.rename_output_layer(output_graph=final_graph, output_node=output_nodes)
            self.__output_layers = g.layers()

        return self.__output_layers

    def freeze(self, out_graph_file=None) -> ((int, int), str):
        """
        This freezes the model and saves it as a graph

        Note, this function will not be needed in its current state when OpenCV 4.4.0 comes out as they will add the ability
        to open keras file (.h5). source: https://github.com/opencv/opencv/issues/16582#issuecomment-630295971

        :param model_file: in the format .h5
        :param out_graph_file: format .pb
        :return: (image_size[1], image_size[2]), input_str
        """

        assert os.path.exists(self._graph_path)
        assert os.path.splitext(self._graph_path)[1] == '.h5'

        # Note: by default the swift activation function (which is used by efficientnet) does not exist.
        # we have to `import efficientnet.tfkeras` to add it if want to load an efficientnet.
        # model = tf.keras.models.load_model(model_file)

        image_size = self.__model.input_shape

        # sort of from: https://github.com/opencv/opencv/issues/16879#issuecomment-603815872
        input_str = 'input'  # it is defined by the lambda
        f = tf.function(lambda input: self.__model(input)).get_concrete_function(
            tf.TensorSpec(shape=list(image_size), dtype=tf.float32))
        f2 = convert_variables_to_constants_v2(f)
        graph_def = f2.graph.as_graph_def()

        # Export frozen graph
        with tf.io.gfile.GFile(out_graph_file, 'wb') as f:
            f.write(graph_def.SerializeToString())

        self.__frozen_model = out_graph_file

        return (image_size[1], image_size[2]), input_str

    def optimize(self, input_node_names, output_node_names, output_path, use_optimize_for_inference_lib=False):
        """
        calls the tensorflow optimize_for_inference python tool through optimize_for_inference_lib

        @param input_node_names:
        @param output_node_names:
        @param output_path:
        @return: output_path
        @param use_optimize_for_inference_lib:
        """
        assert self.__frozen_model
        input_graph_path = self.__frozen_model

        input_graph_def = graph_pb2.GraphDef()
        with tf.io.gfile.GFile(input_graph_path, "rb") as f:
            data = f.read()
            input_graph_def.ParseFromString(data)

        if use_optimize_for_inference_lib:
            output_graph_def = optimize_for_inference_lib.optimize_for_inference(
                input_graph_def, input_node_names.split(","), output_node_names.split(","),
                dtypes.float32.as_datatype_enum)
        else:
            output_graph_def = tf.compat.v1.graph_util.remove_training_nodes(input_graph_def)

        f = tf.io.gfile.GFile(output_path, "w")
        f.write(output_graph_def.SerializeToString())

        self.__optimized_model = output_path
        # return output_path

    def clean_up(self):
        os.remove(self.__frozen_model)
        os.remove(self.__optimized_model)
        os.remove(self._graph_path)

    @staticmethod
    def operationalize(model_file, out_graph_file, input_node_names='input',
                       output_node_names='Identity', clean=True) -> str:
        """
        Takes a keras model file (.h5) and turns it into an operation ready graph that OpenCV can read.
        Calls to_graph_def_protobuf_file, optimize, and transform_graph.
        Note, transform_graph requires WSL.
        Here is a setup guide: `\\qa\tmp\1zjorquera\WSL_tf\set_up.md`.

        @param model_file: in the format .h5
        @param input_node_names: Most likely should be 'input_1'
        @param output_node_names: Most likely should be 'Identity'
        @param out_graph_file: format .pb
        @return: final graph path
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

        # 1. Freeze
        img_shape, input_node_str = keras_model.freeze(freeze_output)

        if output_node_names not in keras_model.output_layers:
            for i in range(len(keras_model.output_layers)):
                print(f"{i} - {keras_model.output_layers[i]}")

            output_node_names = keras_model.output_layers[eval(input("Select an index for inference layer: "))]

        # 2 Optimize
        keras_model.optimize(input_node_names, output_node_names, optimize_output, True)

        transforms = ['remove_nodes(op=PlaceholderWithDefault)',
                      'strip_unused_nodes(type=float, '
                      'shape="1,{},{},3")'.format(img_shape[0], img_shape[1]),
                      'sort_by_execution_order']
        # 3 Transform
        if transform_graph(optimize_output, input_node_names, output_node_names,
                           transforms, out_graph_file) is None:
            print('Failed to call transform_graph through wsl: skipping step.')

        if clean:
            keras_model.clean_up()

        return out_graph_file, output_node_names

# if __name__ == '__main__':
#     final_graph = r"E:\viNet_RnD\Research\Candidates\GWA\B1\efficientnetB0_epoch1_Goldwind.RnD.Operational.pb"
#     input_graph = r"E:\viNet_RnD\Research\Candidates\GWA\B1\viNet_3.0_B1_done.ckpt_GWA.RnD.pb"
#     output_nodes = 'efficientnetb1/predictions/Softmax'
#     kg = KerasModelWrapper(r"E:\viNet_RnD\Research\Candidates\GWA\B1\B1_done.ckpt_GWA.h5")
#     kg.freeze(out_graph_file=r"E:\viNet_RnD\Research\Candidates\GWA\B1\B1_done.ckpt_GWA.Frozen.pb")
#     g = viNetProtoBufWrapper(kg.frozen)
#     # g.rename_output_layer(output_graph=final_graph, output_node=output_nodes)
#     g.inference_layers()
