import argparse
import glob
import os
import shutil

from core.api.nn_graph_api import KerasModelWrapper as kmw, ProtobufWrapper
from customer.customers import CustomerObjectMap
from core.database import database_interface as dbi
from core.deployment.inference import InferenceModel
from neuralnet import NeuralNetConfig, NeuralNet
from application.viNet import viNet
from common.io.interactive import query_tag, query_class_group
from common.audit import path_exists
from common.db.stored_procedures import StoredProcedures as sp


parser = argparse.ArgumentParser(description='One-Click Deployment CLI (viAi)')

parser.add_argument('--keras_model_path',
                    default=None, help="Absolute path to .h5 model file")
parser.add_argument('--output_dir',
                    default=r'E:\viNet_RnD\Research\Candidates',
                    help='Operational model gets saved here')
parser.add_argument('--output_node',
                    default=None, help="Name of output node")
parser.add_argument('--input_node', default='input', help='Name of input node')
parser.add_argument('--model_name',
                    default='viNet_{}_{}_{}_{}.RnD.pb',
                    help='File name for the operational model. '
                         'viNet_<version_major>.<version_minor>_<arch>_<unique_info>_<customer>.<purpose>.pb')
parser.add_argument('--from_checkpoint',
                    default=True,
                    help="Set to true if network is to be operationalized from checkpoints")
parser.add_argument('--checkpoint_path', default=r'E:\viNet_RnD\Research\GWA\MobileNetV2',
                    help='Path to checkpoint files. Used when from_checkpoint is set to true')
parser.add_argument('--renamed',
                    default='viNet_{}_{}_{}_{}.RnD.Renamed.pb')
parser.add_argument('--customer', default='GWA', help="Customer")
parser.add_argument('--checkpoint', default=None,
                    help="The checkpoint to use when operationalizing from checkpoints. "
                         "If not supplied, defaults to latest checkpoint")
parser.add_argument('--net', default="Xception", help="The base architecture")
parser.add_argument('--version', default=3.0, help="Framework version")
parser.add_argument('--db_config', default=r'C:\ProgramData\viNet\config\database.ini',
                    help="Database config file to create connection string from")
parser.add_argument('--evaluate', default=True, help="Evaluate model on a select tag")


if __name__ == '__main__':

    args = parser.parse_args()
    assert (args.keras_model_path and not args.from_checkpoint) \
           or (not args.keras_model_path and args.from_checkpoint)

    # DB Server
    server = dbi.PgsqlInterface()
    assert path_exists(args.db_config), f'{args.db_config} not found'
    server.connect(args.db_config)

    if args.from_checkpoint:
        """
        Folder structure 
        -- checkpoint_path
             -- Customer 
                    -- Network 
                        -- Timestamps 
                            -- checkpoints
                                    ....     
        """
        assert args.checkpoint_path
        assert path_exists(args.checkpoint_path)
        latest_training = max(glob.glob(args.checkpoint_path + "/*"), key=os.path.getctime)
        checkpoints = os.path.join(latest_training, 'checkpoints')
        assert path_exists(checkpoints), checkpoints

        # 1. Keras model, usable in python
        # grab the latest checkpoint directory - training output
        if not args.checkpoint:
            # grab the latest checkpoint
            weights = max(glob.glob(checkpoints + "/*.data*"), key=os.path.getctime)
            weights = os.path.splitext(weights)[0]
        else:
            weights = os.path.join(checkpoints, args.checkpoint)
        # assert os.path.exists(weights + ), "No checkpoints file found, file: {}".format(weights)

        # Load training config, build keras model
        config_file = os.path.join(latest_training, 'config.ini')
        assert path_exists(config_file)
        customer = CustomerObjectMap[args.customer]
        assert customer
        config = NeuralNetConfig(config_file)
        # config.set_input_shape((32, 32, 3))

        model_dir = os.path.join(args.output_dir, customer().name, config.net)
        model = NeuralNet(config, customer)
        args.model_name = args.model_name.format(args.version, config.net,
                                                 os.path.basename(weights),
                                                 model.customer)
        print("Loading checkpoints, path: {}".format(weights))
        model.load_weights(weights)
        # save model as keras .h5 file
        h5_model_path = model.save(model_dir, epochs=os.path.basename(weights), customer=customer().name)
        assert h5_model_path
        assert path_exists(h5_model_path)
        input_node_names = args.input_node
        output_node_names = None
        # if args.output_node in model.output_nodes:
        output_node_names = args.output_node

        # 2. Inference, runtime model - can be loaded in opencv, openvino
        operational_model_path = os.path.join(model_dir, args.model_name)
        class_map = os.path.splitext(args.model_name)[0] + '_ClassMap.txt'
        train_config = os.path.splitext(args.model_name)[0] + '.ini'
        labels_file = os.path.join(checkpoints, 'labels.txt')
        class_map = os.path.join(model_dir, class_map)
        train_config = os.path.join(model_dir, train_config)
        shutil.copyfile(labels_file, class_map)
        shutil.copyfile(config_file, train_config)

        _, args.output_node = kmw.operationalize(h5_model_path,
                                                 operational_model_path,
                                                 output_node_names=output_node_names,
                                                 clean=True)

        protobuf = ProtobufWrapper(operational_model_path)
        protobuf.rename_output_layer(output_graph=operational_model_path,
                                     output_node=args.output_node)
        assert 'output' in protobuf.layers(num_classes=config.num_classes)
        # verify final model can be loaded with opencv python
        inference_model = InferenceModel(operational_model_path, class_map,
                                         (model.input_shape[0], model.input_shape[0]))
        frame = sp.get_frames('Raven', 1, connection=server.connection)

        # 3. Create candidate model
        candidate = viNet(inference_model, customer().name)
        # check load with opencv
        assert candidate.predict(frame)
        candidate.rename(candidate_index=0)

        # upload to database
        candidate.upload()
        configs = sp.vinet_configs(server.connection)
        assert candidate.config in configs

        if args.evaluate:
            # 4. Evaluate model on a  tag
            print("\n")
            tag = query_tag(server.connection, customer=customer)
            assert candidate.verify(tag)

            # 5. Generate report
            class_group = query_class_group(server)
            filename = os.path.join(model_dir, f"{config}.pdf")
            candidate.generate_report(class_group, model_dir, tag=tag)
