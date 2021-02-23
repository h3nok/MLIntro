from transform.graph_transforms import operationalize
from core.training.checkpoints import Checkpoints
import os
from common.io.database import upload_model_to_db, evaluate_candidate_formal as eval
from common.io.interactive import query_tag, query_config
import database_interface as dbi


class LegacyModel:
    def __init__(self, input_graph, output_path=None,
                 binary=True, input_nodes='input', output_nodes='output',
                 input_size=224, customer=None, net_version='2.9', dataset_version='1'):
        """
        Legacy model interface, Inception V2 with default parameters

        @param input_graph:
        @param output_path:
        @param binary:
        @param input_nodes:
        @param output_nodes:
        @param input_size:
        """

        self._input_graph = input_graph
        self._output_path = output_path
        self._checkpoint = None
        self._checkpoint_dir = None
        self._binary = binary
        self._input_node = input_nodes
        self._output_node = output_nodes
        self._customer = customer
        self._input_size = (input_size, input_size)
        self._net_version = net_version
        self._dataset_version = dataset_version
        self._config = None
        self._candidate_model = None

    def operationalize(self, version=2.9):
        """

        @param version:
        @return:
        """
        checkpoint = Checkpoints(self._customer, vinet_version=version)
        self._checkpoint_dir = checkpoint.path
        self._checkpoint = checkpoint.latest_checkpoint

        model_name = f"viNet_{self._net_version}_{self._customer}" \
                     f"_dataset_v{self._dataset_version}_{checkpoint.iteration}itr.pb "
        if not self._output_path:
            self._output_path = checkpoint.path
        output_graph = os.path.join(self._output_path, model_name)
        self._candidate_model = output_graph

        return operationalize(self._input_graph, self._checkpoint_dir,
                              self._input_node, self._output_node,
                              output_graph, self._input_size,
                              checkpoint_file=self._checkpoint, is_binary=self._binary)

    def upload(self, class_map=None):
        """
        Upload candidate model (legacy) to viNet database

        @param class_map:
        @return:
        """
        binary = self._candidate_model
        if not binary or (not os.path.exists(binary)):
            binary = input("Enter path to network pb file: ")
        assert binary, "Must supply path to binary model file (.pb)"
        print("Candidate model: {}".format(binary))
        if not class_map:
            class_map = (input("Enter path to classification code map: "))
        assert class_map, "Must supply a path to id-to-class map file"
        input_node = self._input_node
        output_node = self._output_node
        mean = input("Enter mean value (default=127): ")
        if mean == '':
            mean = 127
        # assert mean, "Must supply mean value"
        size = self._input_size[0]
        assert size, "Must supply input size"
        self._config = input("Enter config name: ")
        # image_type = "BoundingBox"
        net_name = input("Enter network name: ")
        if not net_name:
            net_name = self._config
        database = "viNet"
        host = "12.1.1.91"

        upload_model_to_db(binary, class_map, size,
                           mean, input_node, output_node,
                           database, host, self._config, net_name)

    def run_verification(self, tag=None, customer=None):
        if not self._customer:
            assert customer, "Must supply customer"
            self._customer = customer
        server = dbi.PgsqlInterface()
        server.connect()

        if not tag:
            tag = query_tag(server.connection, customer=self._customer)
        if not self._config:
            self._config = query_config(server.connection, self._customer)

        eval(self._config, tag)

